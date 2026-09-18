#!/usr/bin/env bash
set -euo pipefail

repository_url="${1:?usage: clean_clone_verify.sh REPOSITORY_URL [WORK_DIR]}"
work_dir="${2:-$(mktemp -d /tmp/robust-apex-clean-clone.XXXXXX)}"
clone_dir="${work_dir}/repository"
submission_branch="${SUBMISSION_BRANCH:-}"
summary_log="${work_dir}/clean_clone.log"

mkdir -p "${work_dir}"
clone_args=(--depth 1)
if [[ -n "${submission_branch}" ]]; then
  clone_args+=(--branch "${submission_branch}" --single-branch)
fi
git clone "${clone_args[@]}" "${repository_url}" "${clone_dir}"
git -C "${clone_dir}" lfs install --local
git -C "${clone_dir}" lfs pull
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync --frozen --project "${clone_dir}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --project "${clone_dir}" generate \
  --output-dir "${work_dir}/full_run_1"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --project "${clone_dir}" generate \
  --output-dir "${work_dir}/full_run_2"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --project "${clone_dir}" verify-local \
  --output-dir "${work_dir}/full_run_1"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --project "${clone_dir}" verify-local \
  --output-dir "${work_dir}/full_run_2"

for artifact in library.fasta top.fasta ranking.tsv; do
  cmp "${work_dir}/full_run_1/${artifact}" "${work_dir}/full_run_2/${artifact}"
done

for output_dir in "${work_dir}/full_run_1" "${work_dir}/full_run_2"; do
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run --project "${clone_dir}" python \
    "${clone_dir}/scripts/verify_existing_output.py" \
    --output-dir "${output_dir}" \
    --antibacterial-fasta "${clone_dir}/data/antibacterial.fasta"
done

{
  echo "status=passed"
  echo "repository_url=${repository_url}"
  echo "submission_branch=${submission_branch:-default}"
  echo "commit=$(git -C "${clone_dir}" rev-parse HEAD)"
  sha256sum \
    "${work_dir}/full_run_1/library.fasta" \
    "${work_dir}/full_run_1/top.fasta" \
    "${work_dir}/full_run_1/ranking.tsv" \
    "${work_dir}/full_run_1/manifest.json" \
    "${work_dir}/full_run_2/library.fasta" \
    "${work_dir}/full_run_2/top.fasta" \
    "${work_dir}/full_run_2/ranking.tsv" \
    "${work_dir}/full_run_2/manifest.json"
} >"${summary_log}"

echo "Clean-clone verification passed; artifacts are under ${work_dir}"
echo "Summary log: ${summary_log}"
