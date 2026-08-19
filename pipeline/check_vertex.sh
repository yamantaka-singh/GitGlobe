#!/usr/bin/env bash
# Is the gitglobe billing account open, and does Vertex actually answer?
# Run after reactivating billing in the Console:
#   ./check_vertex.sh
set -euo pipefail

echo "--- billing account state ---"
gcloud billing accounts describe 014FBF-4E09C8-4842DC --format="value(open)"

echo "--- project -> billing link ---"
gcloud billing projects describe gitglobe --format="value(billingEnabled)"

echo "--- actual Vertex call ---"
TOKEN=$(gcloud auth print-access-token)
curl -s -o /dev/null -w "HTTP %{http_code}\n" \
  "https://us-central1-aiplatform.googleapis.com/v1/projects/gitglobe/locations/us-central1/publishers/google/models/gemini-embedding-001:predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"content":"test","task_type":"CLUSTERING"}],"parameters":{"outputDimensionality":768}}'
