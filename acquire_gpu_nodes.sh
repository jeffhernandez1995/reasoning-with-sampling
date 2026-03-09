#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   PROJECT=tti-rava-vicenteor ./acquire_gpu_nodes.sh
# Optional overrides:
#   H200_ENABLED=true
#   H200_RESERVATION=projects/PROJECT/reservations/RESERVATION_NAME
#   H200_RESERVATION_ZONE=us-central1-a
#   H200_INITIAL_REGION=us-central1
#   NETWORK_NAME=gpu
#   SUBNETWORK_NAME=gpu
#   A100_COUNT=5
#   A100_ZONES_CSV=us-central1-a,us-central1-c

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

is_truthy() {
  case "$1" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

for cmd in gcloud awk sort head seq sed wc tr; do
  require_cmd "$cmd"
done

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
[[ -n "${PROJECT}" && "${PROJECT}" != "(unset)" ]] || die "Set PROJECT=... or run: gcloud config set project YOUR_PROJECT"

IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-200}"

SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
PER_ZONE_PAUSE_SECONDS="${PER_ZONE_PAUSE_SECONDS:-5}"
DELETE_POLL_SECONDS="${DELETE_POLL_SECONDS:-5}"
DELETE_WAIT_SECONDS="${DELETE_WAIT_SECONDS:-300}"

H200_NAME="${H200_NAME:-h200-8g-01}"
H200_MACHINE_TYPE="a3-ultragpu-8g"
H200_ENABLED="${H200_ENABLED:-false}"
H200_RESERVATION="${H200_RESERVATION:-}"
H200_RESERVATION_ZONE="${H200_RESERVATION_ZONE:-}"
H200_INITIAL_REGION="${H200_INITIAL_REGION:-us-central1}"
H200_TERMINATION_ACTION="${H200_TERMINATION_ACTION:-STOP}"

NETWORK_NAME="${NETWORK_NAME:-}"
SUBNETWORK_NAME="${SUBNETWORK_NAME:-}"

A100_PREFIX="${A100_PREFIX:-a100-8g}"
A100_MACHINE_TYPE="a2-ultragpu-8g"
A100_COUNT="${A100_COUNT:-5}"
A100_PROVISIONING_MODEL="${A100_PROVISIONING_MODEL:-STANDARD}"
A100_ZONES_CSV="${A100_ZONES_CSV:-}"

LAST_TRY_OUTPUT=""
A100_NAME_REGEX="^${A100_PREFIX}-[0-9][0-9]$"

[[ -n "$NETWORK_NAME" || -z "$SUBNETWORK_NAME" ]] || die "SUBNETWORK_NAME requires NETWORK_NAME"

if is_truthy "$H200_ENABLED"; then
  H200_ENABLED=1
else
  H200_ENABLED=0
fi

is_quota_error() {
  grep -Eqi 'quota|QUOTA_EXCEEDED|insufficient.*quota|Quota .* exceeded|GPUS_ALL_REGIONS|CPUS(_ALL_REGIONS)?' <<<"$1"
}

is_capacity_error() {
  grep -Eqi 'ZONE_RESOURCE_POOL_EXHAUSTED|ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS|does not have enough resources|currently unavailable|resource availability|stockout|capacity|available resources in reservation|reservation.*exhausted|insufficient reservation' <<<"$1"
}

print_rotated_args() {
  local n="$#"
  local offset i idx
  local args=("$@")

  [[ "$n" -eq 0 ]] && return 0

  offset=$(( $(date +%s) % n ))
  for ((i=0; i<n; i++)); do
    idx=$(( (offset + i) % n ))
    printf '%s\n' "${args[$idx]}"
  done
}

list_zones_for_machine_type() {
  local machine_type="$1"

  gcloud compute machine-types list \
    --project="$PROJECT" \
    --filter="name=('${machine_type}')" \
    --format='value(zone)' \
  | awk -F/ 'NF {print $NF}' \
  | sort -u
}

load_a100_zones() {
  if [[ -n "$A100_ZONES_CSV" ]]; then
    tr ',' '\n' <<<"$A100_ZONES_CSV" | sed '/^$/d' | sort -u
    return 0
  fi

  list_zones_for_machine_type "$A100_MACHINE_TYPE"
}

zone_to_region() {
  local zone="$1"
  printf '%s\n' "${zone%-*}"
}

default_network_exists() {
  gcloud compute networks describe default --project="$PROJECT" >/dev/null 2>&1
}

find_subnet_for_network_in_region() {
  local network="$1"
  local region="$2"

  gcloud compute networks subnets list \
    --project="$PROJECT" \
    --regions="$region" \
    --filter="network.basename()=${network}" \
    --format='value(name)' \
    --limit=1 2>/dev/null || true
}

find_any_subnet_in_region() {
  local region="$1"

  gcloud compute networks subnets list \
    --project="$PROJECT" \
    --regions="$region" \
    --format='csv[no-heading](name,network.basename())' \
    --limit=1 2>/dev/null || true
}

network_interface_spec_for_zone() {
  local zone="$1"
  local extra="${2:-}"
  local region subnet_line subnet network spec

  region="$(zone_to_region "$zone")"

  if [[ -n "$NETWORK_NAME" && -n "$SUBNETWORK_NAME" ]]; then
    if gcloud compute networks subnets describe "$SUBNETWORK_NAME" \
      --project="$PROJECT" \
      --region="$region" >/dev/null 2>&1; then
      network="$NETWORK_NAME"
      subnet="$SUBNETWORK_NAME"
    else
      return 1
    fi
  elif [[ -n "$NETWORK_NAME" ]]; then
    if [[ "$NETWORK_NAME" == "default" ]] && default_network_exists; then
      spec="network=default"
      [[ -n "$extra" ]] && spec+=",${extra}"
      printf '%s\n' "$spec"
      return 0
    fi

    subnet="$(find_subnet_for_network_in_region "$NETWORK_NAME" "$region")"
    [[ -n "$subnet" ]] || return 1
    network="$NETWORK_NAME"
  else
    if default_network_exists; then
      spec="network=default"
      [[ -n "$extra" ]] && spec+=",${extra}"
      printf '%s\n' "$spec"
      return 0
    fi

    subnet_line="$(find_any_subnet_in_region "$region")"
    [[ -n "$subnet_line" ]] || return 1
    subnet="${subnet_line%%,*}"
    network="${subnet_line##*,}"
  fi

  spec="network=projects/${PROJECT}/global/networks/${network},subnet=projects/${PROJECT}/regions/${region}/subnetworks/${subnet}"
  [[ -n "$extra" ]] && spec+=",${extra}"
  printf '%s\n' "$spec"
}

zone_has_supported_network() {
  network_interface_spec_for_zone "$1" "${2:-}" >/dev/null 2>&1
}

instance_info() {
  local name="$1"
  local lines line_count first_line zone_url status zone

  lines="$(gcloud compute instances list \
    --project="$PROJECT" \
    --filter="name~'^${name}$'" \
    --format='csv[no-heading](zone,status)' 2>/dev/null || true)"

  line_count="$(printf '%s\n' "$lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  if [[ "$line_count" -eq 0 ]]; then
    return 1
  fi

  if [[ "$line_count" -gt 1 ]]; then
    printf '%s\n' "$lines" >&2
    die "Multiple instances found with name ${name}; use unique names before rerunning."
  fi

  first_line="$(printf '%s\n' "$lines" | head -n1)"
  zone_url="${first_line%%,*}"
  status="${first_line##*,}"
  zone="${zone_url##*/}"

  printf '%s %s\n' "$zone" "$status"
}

instance_is_ready() {
  local name="$1"
  local info status

  info="$(instance_info "$name" || true)"
  [[ -n "$info" ]] || return 1

  status="${info##* }"
  case "$status" in
    RUNNING|PROVISIONING|STAGING) return 0 ;;
    *) return 1 ;;
  esac
}

start_existing_instance() {
  local name="$1"
  local zone="$2"
  local out rc

  log "Trying to start existing instance ${name} in ${zone}"

  set +e
  out="$(gcloud compute instances start "$name" --project="$PROJECT" --zone="$zone" 2>&1)"
  rc=$?
  set -e

  LAST_TRY_OUTPUT="$out"
  if [[ $rc -eq 0 ]]; then
    log "Started ${name} in ${zone}"
    return 0
  fi

  printf '%s\n' "$out" >&2
  return $rc
}

create_h200_in_zone() {
  local zone="$1"
  local reservation_ref="$2"
  local region
  local network_interface
  local out rc

  region="$(zone_to_region "$zone")"
  network_interface="$(network_interface_spec_for_zone "$zone" "nic-type=GVNIC" || true)"
  [[ -n "$network_interface" ]] || die "No usable network/subnet found for H200 zone ${zone}. Set NETWORK_NAME/SUBNETWORK_NAME or add a subnet in ${region}."

  local cmd=(
    gcloud compute instances create "$H200_NAME"
    --project="$PROJECT"
    --zone="$zone"
    --machine-type="$H200_MACHINE_TYPE"
    --image-family="$IMAGE_FAMILY"
    --image-project="$IMAGE_PROJECT"
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB"
    --boot-disk-type=hyperdisk-balanced
    --maintenance-policy=TERMINATE
    --network-interface="$network_interface"
    --reservation-affinity=specific
    --reservation="$reservation_ref"
    --provisioning-model=RESERVATION_BOUND
    --instance-termination-action="$H200_TERMINATION_ACTION"
    --restart-on-failure
    --scopes=https://www.googleapis.com/auth/cloud-platform
  )

  log "Trying H200 reservation-bound instance in ${zone} with reservation ${reservation_ref}"

  set +e
  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  set -e

  LAST_TRY_OUTPUT="$out"
  if [[ $rc -eq 0 ]]; then
    log "Created H200 node in ${zone}"
    return 0
  fi

  printf '%s\n' "$out" >&2
  return $rc
}

create_a100_in_zone() {
  local name="$1"
  local zone="$2"
  local region
  local network_interface
  local out rc

  region="$(zone_to_region "$zone")"
  network_interface="$(network_interface_spec_for_zone "$zone" || true)"
  [[ -n "$network_interface" ]] || die "No usable network/subnet found for A100 zone ${zone}. Set NETWORK_NAME/SUBNETWORK_NAME or add a subnet in ${region}."

  local cmd=(
    gcloud compute instances create "$name"
    --project="$PROJECT"
    --zone="$zone"
    --machine-type="$A100_MACHINE_TYPE"
    --image-family="$IMAGE_FAMILY"
    --image-project="$IMAGE_PROJECT"
    --boot-disk-size="${BOOT_DISK_SIZE_GB}GB"
    --boot-disk-type=pd-balanced
    --maintenance-policy=TERMINATE
    --network-interface="$network_interface"
    --restart-on-failure
    --provisioning-model="$A100_PROVISIONING_MODEL"
    --scopes=https://www.googleapis.com/auth/cloud-platform
  )

  log "Trying A100 node ${name} in ${zone}"

  set +e
  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  set -e

  LAST_TRY_OUTPUT="$out"
  if [[ $rc -eq 0 ]]; then
    log "Created A100 node ${name} in ${zone}"
    return 0
  fi

  printf '%s\n' "$out" >&2
  return $rc
}

handle_failure() {
  local context="$1"

  if is_quota_error "$LAST_TRY_OUTPUT"; then
    die "${context} failed due to quota or quota-like limits. Fix quota and rerun."
  fi

  if is_capacity_error "$LAST_TRY_OUTPUT"; then
    return 1
  fi

  die "${context} failed with a non-capacity error. See output above."
}

wait_for_instances_absent() {
  local deadline now name
  local names=("$@")

  deadline=$(( $(date +%s) + DELETE_WAIT_SECONDS ))

  while true; do
    local remaining=0

    for name in "${names[@]+${names[@]}}"; do
      [[ -n "$name" ]] || continue
      if instance_info "$name" >/dev/null 2>&1; then
        remaining=1
        break
      fi
    done

    if [[ "$remaining" -eq 0 ]]; then
      return 0
    fi

    now=$(date +%s)
    if (( now >= deadline )); then
      die "Timed out waiting for instance deletions to finish: ${names[*]}"
    fi

    sleep "$DELETE_POLL_SECONDS"
  done
}

delete_instances_best_effort() {
  local zone="$1"
  shift || true
  local name
  local names=("$@")

  [[ "$#" -gt 0 ]] || return 0

  for name in "${names[@]+${names[@]}}"; do
    [[ -n "$name" ]] || continue
    log "Deleting partial instance ${name} in ${zone}"
    gcloud compute instances delete "$name" \
      --project="$PROJECT" \
      --zone="$zone" \
      --quiet >/dev/null 2>&1 || true
  done

  wait_for_instances_absent "${names[@]+${names[@]}}"
}

reservation_name_from_ref() {
  local ref="$1"

  if [[ "$ref" == *"/reservationBlocks/"* ]]; then
    ref="${ref%%/reservationBlocks/*}"
  fi
  printf '%s\n' "${ref##*/}"
}

list_h200_reservation_rows() {
  gcloud compute reservations list \
    --project="$PROJECT" \
    --format='csv[no-heading](name,zone,status,specificReservation.instanceProperties.machineType,specificReservation.count,specificReservation.inUseCount,selfLink)' 2>/dev/null || true
}

resolve_h200_reservations() {
  local target_name zone_hint
  local row_name row_zone row_status row_machine row_count row_used row_link zone
  local found=0

  if [[ -n "$H200_RESERVATION" ]]; then
    target_name="$(reservation_name_from_ref "$H200_RESERVATION")"
    zone_hint="$H200_RESERVATION_ZONE"

    while IFS=, read -r row_name row_zone row_status row_machine row_count row_used row_link; do
      [[ -n "$row_name" ]] || continue
      [[ "$row_machine" == "$H200_MACHINE_TYPE" ]] || continue
      zone="${row_zone##*/}"
      if [[ "$row_name" == "$target_name" || "$row_link" == "$H200_RESERVATION" ]]; then
        printf '%s|%s|%s|%s|%s|%s\n' "$H200_RESERVATION" "$zone" "$row_status" "$row_count" "$row_used" "$row_name"
        found=1
      fi
    done < <(list_h200_reservation_rows)

    if [[ "$found" -eq 1 ]]; then
      return 0
    fi

    if [[ -n "$zone_hint" ]]; then
      printf '%s|%s|UNKNOWN|0|0|%s\n' "$H200_RESERVATION" "$zone_hint" "$target_name"
      return 0
    fi

    die "Could not resolve H200 reservation '${H200_RESERVATION}' to a zone. Set H200_RESERVATION_ZONE=... or use a reservation visible to gcloud."
  fi

  while IFS=, read -r row_name row_zone row_status row_machine row_count row_used row_link; do
    [[ -n "$row_name" ]] || continue
    [[ "$row_machine" == "$H200_MACHINE_TYPE" ]] || continue
    zone="${row_zone##*/}"
    printf '%s|%s|%s|%s|%s|%s\n' "$row_name" "$zone" "$row_status" "$row_count" "$row_used" "$row_name"
    found=1
  done < <(list_h200_reservation_rows)

  [[ "$found" -eq 1 ]] || die "No ${H200_MACHINE_TYPE} reservations found in project ${PROJECT}. Set H200_RESERVATION=... or create reservation-backed H200 capacity first."
}

try_h200_candidates() {
  local label="$1"
  shift || true
  local candidate ref cand_zone cand_status cand_count cand_used cand_name
  local attempted_any=0

  if [[ "$#" -eq 0 ]]; then
    log "No H200 reservation candidates available for ${label}"
    return 0
  fi

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue

    IFS='|' read -r ref cand_zone cand_status cand_count cand_used cand_name <<<"$candidate"
    log "Considering H200 reservation ${cand_name} in ${cand_zone} (status=${cand_status}, count=${cand_count}, in_use=${cand_used}) for ${label}"

    case "$cand_status" in
      READY|UNKNOWN|'')
        attempted_any=1
        ;;
      *)
        log "Skipping reservation ${cand_name} in ${cand_zone} until it becomes READY"
        continue
        ;;
    esac

    if create_h200_in_zone "$cand_zone" "$ref"; then
      return 0
    fi

    if handle_failure "Creating ${H200_NAME} in ${cand_zone}"; then
      :
    else
      log "No H200 reservation capacity available in ${cand_zone}; trying next reservation candidate"
      sleep "$PER_ZONE_PAUSE_SECONDS"
    fi
  done < <(print_rotated_args "$@")

  if [[ "$attempted_any" -eq 0 ]]; then
    log "No READY H200 reservations are available yet for ${label}; waiting for the next retry."
  fi

  return 0
}

try_h200_once_in_region() {
  local region="$1"
  local candidate cand_zone
  local candidates=()
  local regional_candidates=()
  local regional_candidate_count=0

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && candidates+=("$candidate")
  done < <(resolve_h200_reservations)

  for candidate in "${candidates[@]+${candidates[@]}}"; do
    IFS='|' read -r _ cand_zone _ _ _ _ <<<"$candidate"
    if [[ "$cand_zone" == "${region}-"* ]] && zone_has_supported_network "$cand_zone" "nic-type=GVNIC"; then
      regional_candidates+=("$candidate")
      regional_candidate_count=$((regional_candidate_count + 1))
    fi
  done

  if [[ "$regional_candidate_count" -eq 0 ]]; then
    log "No H200 reservation candidates with usable networking found in ${region}; skipping the initial one-shot attempt"
    return 0
  fi

  log "Initial H200 one-shot attempt in ${region} before entering the retry loop"
  try_h200_candidates "initial ${region} attempt" "${regional_candidates[@]+${regional_candidates[@]}}"
}

ensure_h200() {
  local info zone status
  local region
  local candidates=()
  local candidate_count=0

  info="$(instance_info "$H200_NAME" || true)"
  if [[ -n "$info" ]]; then
    zone="${info%% *}"
    status="${info##* }"

    case "$status" in
      RUNNING|PROVISIONING|STAGING)
        log "${H200_NAME} already ${status} in ${zone}"
        return 0
        ;;
      TERMINATED)
        if start_existing_instance "$H200_NAME" "$zone"; then
          return 0
        fi
        if is_capacity_error "$LAST_TRY_OUTPUT"; then
          log "Could not restart ${H200_NAME} in ${zone} yet; reservation capacity is still unavailable"
          return 0
        fi
        handle_failure "Restarting ${H200_NAME}" || true
        return 0
        ;;
      STOPPING|REPAIRING|SUSPENDING|SUSPENDED)
        log "${H200_NAME} currently ${status} in ${zone}; leaving it alone"
        return 0
        ;;
      *)
        log "${H200_NAME} exists with status ${status} in ${zone}; leaving it alone"
        return 0
        ;;
    esac
  fi

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    IFS='|' read -r _ zone _ _ _ _ <<<"$candidate"
    if ! zone_has_supported_network "$zone" "nic-type=GVNIC"; then
      region="$(zone_to_region "$zone")"
      log "Skipping H200 candidate zone ${zone}; no usable subnet found in ${region}"
      continue
    fi
    candidates+=("$candidate")
    candidate_count=$((candidate_count + 1))
  done < <(resolve_h200_reservations)

  [[ "$candidate_count" -gt 0 ]] || die "No H200 reservation candidates with usable networking are available."
  try_h200_candidates "global retry loop" "${candidates[@]+${candidates[@]}}"
}

list_a100_instances() {
  gcloud compute instances list \
    --project="$PROJECT" \
    --filter="name~'${A100_NAME_REGEX}'" \
    --format='csv[no-heading](name,zone,status)' 2>/dev/null || true
}

try_a100_group_in_empty_zone() {
  local zone="$1"
  local group_names=()
  local i name

  log "Trying to place all ${A100_COUNT} A100 nodes in ${zone}"

  for i in $(seq 1 "$A100_COUNT"); do
    name="$(printf '%s-%02d' "$A100_PREFIX" "$i")"
    group_names+=("$name")

    if create_a100_in_zone "$name" "$zone"; then
      continue
    fi

    log "Zone ${zone} could not fit the full A100 group; rolling back partial creations"
    delete_instances_best_effort "$zone" "${group_names[@]}"

    if handle_failure "Creating ${name} in ${zone}"; then
      :
    else
      return 1
    fi
  done

  log "Placed all ${A100_COUNT} A100 nodes in ${zone}"
  return 0
}

reconcile_a100_group_in_zone() {
  local zone="$1"
  local i name info existing_zone status

  log "Reconciling A100 group in anchored zone ${zone}"

  for i in $(seq 1 "$A100_COUNT"); do
    name="$(printf '%s-%02d' "$A100_PREFIX" "$i")"
    info="$(instance_info "$name" || true)"

    if [[ -n "$info" ]]; then
      existing_zone="${info%% *}"
      status="${info##* }"

      [[ "$existing_zone" == "$zone" ]] || die "${name} exists in ${existing_zone}, not ${zone}. Delete or rename the group before rerunning."

      case "$status" in
        RUNNING|PROVISIONING|STAGING)
          log "${name} already ${status} in ${zone}"
          continue
          ;;
        TERMINATED)
          if start_existing_instance "$name" "$zone"; then
            continue
          fi
          if is_capacity_error "$LAST_TRY_OUTPUT"; then
            log "Could not restart ${name} in ${zone} yet; capacity still unavailable"
            return 1
          fi
          handle_failure "Restarting ${name}" || true
          return 1
          ;;
        STOPPING|REPAIRING|SUSPENDING|SUSPENDED)
          log "${name} currently ${status} in ${zone}; waiting"
          return 1
          ;;
        *)
          log "${name} has status ${status} in ${zone}; waiting"
          return 1
          ;;
      esac
    fi

    if create_a100_in_zone "$name" "$zone"; then
      continue
    fi

    if handle_failure "Creating ${name} in ${zone}"; then
      :
    else
      log "Zone ${zone} still cannot host the full A100 group"
      return 1
    fi
  done

  return 0
}

ensure_a100_group_same_zone() {
  local existing zones_count preferred_zone
  local region
  local zone
  local zones=()
  local zone_count=0

  existing="$(list_a100_instances)"
  if [[ -n "$existing" ]]; then
    zones_count="$(awk -F, 'NF {print $2}' <<<"$existing" | awk -F/ 'NF {print $NF}' | sort -u | wc -l | tr -d ' ')"
    if [[ "$zones_count" -gt 1 ]]; then
      printf '%s\n' "$existing" >&2
      die "Existing A100 nodes are split across multiple zones. Delete them first, or change A100_PREFIX."
    fi

    preferred_zone="$(awk -F, 'NF {print $2}' <<<"$existing" | head -n1)"
    preferred_zone="${preferred_zone##*/}"
    [[ -n "$preferred_zone" ]] || die "Could not determine the existing A100 zone from gcloud output."

    reconcile_a100_group_in_zone "$preferred_zone" || true
    return 0
  fi

  while IFS= read -r zone; do
    [[ -n "$zone" ]] || continue
    if ! zone_has_supported_network "$zone"; then
      region="$(zone_to_region "$zone")"
      log "Skipping A100 candidate zone ${zone}; no usable subnet found in ${region}"
      continue
    fi
    zones+=("$zone")
    zone_count=$((zone_count + 1))
  done < <(load_a100_zones)

  [[ "$zone_count" -gt 0 ]] || die "No zones found for ${A100_MACHINE_TYPE} with usable networking."

  while IFS= read -r zone; do
    [[ -z "$zone" ]] && continue

    if try_a100_group_in_empty_zone "$zone"; then
      return 0
    fi

    log "Zone ${zone} cannot currently host the full A100 group"
    sleep "$PER_ZONE_PAUSE_SECONDS"
  done < <(print_rotated_args "${zones[@]+${zones[@]}}")

  return 0
}

count_ready_a100s() {
  local i name count
  count=0

  for i in $(seq 1 "$A100_COUNT"); do
    name="$(printf '%s-%02d' "$A100_PREFIX" "$i")"
    if instance_is_ready "$name"; then
      count=$((count + 1))
    fi
  done

  printf '%s\n' "$count"
}

A100_ZONES=()
A100_ZONE_COUNT=0
while IFS= read -r z; do
  [[ -n "$z" ]] || continue
  if ! zone_has_supported_network "$z"; then
    continue
  fi
  A100_ZONES+=("$z")
  A100_ZONE_COUNT=$((A100_ZONE_COUNT + 1))
done < <(load_a100_zones)

[[ "$A100_ZONE_COUNT" -gt 0 ]] || die "No zones found for ${A100_MACHINE_TYPE} with usable networking."

log "Project: ${PROJECT}"
log "A100 candidate zones: ${A100_ZONES[*]}"
if [[ -n "$NETWORK_NAME" ]]; then
  log "Network override: ${NETWORK_NAME}${SUBNETWORK_NAME:+ / ${SUBNETWORK_NAME}}"
else
  log "Network override: auto-detect per-zone subnet (default VPC if present, otherwise first subnet in region)"
fi
if [[ "$H200_ENABLED" -eq 1 && -n "$H200_RESERVATION" ]]; then
  log "H200 reservation override: ${H200_RESERVATION}"
elif [[ "$H200_ENABLED" -eq 1 ]]; then
  log "H200 reservation override: auto-discover ${H200_MACHINE_TYPE} reservations"
else
  log "H200 acquisition: disabled"
fi
if [[ "$H200_ENABLED" -eq 1 ]]; then
  log "H200 initial region preference: ${H200_INITIAL_REGION}"
  log "Target: 1 x ${H200_MACHINE_TYPE} (RESERVATION_BOUND) and ${A100_COUNT} x ${A100_MACHINE_TYPE} (${A100_PROVISIONING_MODEL}) with all A100s in one zone"
else
  log "Target: ${A100_COUNT} x ${A100_MACHINE_TYPE} (${A100_PROVISIONING_MODEL}) with all A100s in one zone"
fi

if [[ "$H200_ENABLED" -eq 1 ]] && ! instance_info "$H200_NAME" >/dev/null 2>&1; then
  try_h200_once_in_region "$H200_INITIAL_REGION"
fi

while true; do
  if [[ "$H200_ENABLED" -eq 1 ]]; then
    ensure_h200
  fi
  ensure_a100_group_same_zone

  H200_READY=1
  if [[ "$H200_ENABLED" -eq 1 ]]; then
    H200_READY=0
    if instance_is_ready "$H200_NAME"; then
      H200_READY=1
    fi
  fi

  A100_READY="$(count_ready_a100s)"

  if [[ "$H200_READY" -eq 1 && "$A100_READY" -eq "$A100_COUNT" ]]; then
    if [[ "$H200_ENABLED" -eq 1 ]]; then
      log "Success: all requested nodes are present."
      gcloud compute instances list \
        --project="$PROJECT" \
        --filter="name~'(^${H200_NAME}$|${A100_NAME_REGEX})'" \
        --format='table(name,zone.basename(),status,machineType.basename())'
    else
      log "Success: all requested A100 nodes are present."
      gcloud compute instances list \
        --project="$PROJECT" \
        --filter="name~'${A100_NAME_REGEX}'" \
        --format='table(name,zone.basename(),status,machineType.basename())'
    fi
    exit 0
  fi

  if [[ "$H200_ENABLED" -eq 1 ]]; then
    log "Still waiting. H200 ready=${H200_READY}, A100 ready=${A100_READY}/${A100_COUNT}. Sleeping ${SLEEP_SECONDS}s."
  else
    log "Still waiting. A100 ready=${A100_READY}/${A100_COUNT}. Sleeping ${SLEEP_SECONDS}s."
  fi
  sleep "$SLEEP_SECONDS"
done
