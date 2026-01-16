param(
  [string]$EsUrl  = $env:ES_URL,
  [string]$Index  = $env:INDEX,
  [string]$EsUser = $env:ES_USER,
  [string]$EsPass = $env:ES_PASS,
  [switch]$Insecure
)

$ErrorActionPreference = "Stop"

# Defaults
if ([string]::IsNullOrWhiteSpace($EsUrl))  { $EsUrl = "https://192.168.100.100:9200" }
if ([string]::IsNullOrWhiteSpace($Index))  { $Index = "rankeval_demo" }
if ([string]::IsNullOrWhiteSpace($EsUser)) { $EsUser = "elastic" }
if ([string]::IsNullOrWhiteSpace($EsPass)) { $EsPass = "wnyt2024!!" }

Write-Host "Using ES_URL=[$EsUrl] INDEX=[$Index] USER=[$EsUser] PASS_LEN=[$($EsPass.Length)] Insecure=[$Insecure]"

# curl options (use curl.exe, not PowerShell alias)
$curlCommon = @("-sS", "-u", "$EsUser`:$EsPass")
if ($Insecure) {
  $curlCommon += @("-k", "--ssl-no-revoke")
}

function CurlHeadExists([string]$url) {
  $args = @($curlCommon + @("-o", "NUL", "-w", "%{http_code}", "-I", $url))
  $code = & curl.exe @args
  if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): HEAD $url" }
  return ($code -eq "200")
}

# [1/3] Create index only if missing
if (CurlHeadExists "$EsUrl/$Index") {
  Write-Host "[1/3] Index already exists, skip create: $Index"
} else {
  Write-Host "[1/3] Create index: $Index"
  $body = @"
{
  "mappings": {
    "properties": {
      "title":   { "type": "text" },
      "content": { "type": "text" }
    }
  }
}
"@
  $args = @($curlCommon + @(
    "-X","PUT","$EsUrl/$Index",
    "-H","Content-Type: application/json",
    "--data-binary",$body
  ))
  & curl.exe @args | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): PUT $EsUrl/$Index" }
}

# [2/3] Bulk insert sample docs
Write-Host "[2/3] Bulk insert sample docs"
$bulk = @"
{ "index": { "_index": "$Index", "_id": "doc1" } }
{ "title": "Dream big", "content": "A short story about a dream and ambition. Big goals and persistence." }
{ "index": { "_index": "$Index", "_id": "doc2" } }
{ "title": "Night thoughts", "content": "I had a vivid dream last night. Not about ambition, just random scenes." }
{ "index": { "_index": "$Index", "_id": "doc3" } }
{ "title": "Dream Theater setlist", "content": "Concert review and setlist. Music band named Dream Theater." }
{ "index": { "_index": "$Index", "_id": "doc4" } }
{ "title": "Ambition and goals", "content": "How to set goals. Dreaming is not enough without a plan." }
{ "index": { "_index": "$Index", "_id": "doc5" } }
{ "title": "SAN GIORGIO equipment manual", "content": "Installation instructions for SAN GIORGIO pump equipment. Safety notes." }
{ "index": { "_index": "$Index", "_id": "doc6" } }
{ "title": "SAN GIORGIO troubleshooting guide", "content": "Manual-like troubleshooting steps. Error codes and maintenance." }
{ "index": { "_index": "$Index", "_id": "doc7" } }
{ "title": "Giorgio parts catalog", "content": "Replacement parts for San Giorgio equipment. Part numbers and diagrams." }
{ "index": { "_index": "$Index", "_id": "doc8" } }
{ "title": "San Giorgio restaurant review", "content": "A restaurant called San Giorgio. Menu and reviews." }
{ "index": { "_index": "$Index", "_id": "doc9" } }
{ "title": "Giorgio Armani new collection", "content": "Fashion news about Giorgio Armani. Not related to equipment." }
{ "index": { "_index": "$Index", "_id": "doc10" } }
{ "title": "Elasticsearch rank eval guide", "content": "Using _rank_eval to evaluate search relevance. Precision, recall, DCG, MRR." }
{ "index": { "_index": "$Index", "_id": "doc11" } }
{ "title": "Search templates with mustache", "content": "How to use mustache search templates. Render with _render/template." }
{ "index": { "_index": "$Index", "_id": "doc12" } }
{ "title": "Ranking evaluation metrics", "content": "DCG, NDCG, MRR, precision@k explained with examples." }
{ "index": { "_index": "$Index", "_id": "doc13" } }
{ "title": "Elasticsearch query tuning", "content": "Boosting title vs content fields. match operator and minimum_should_match." }
{ "index": { "_index": "$Index", "_id": "doc14" } }
{ "title": "Kubernetes basics", "content": "Pods, deployments, services. Quickstart for beginners." }
{ "index": { "_index": "$Index", "_id": "doc15" } }
{ "title": "K8s networking deep dive", "content": "DNS, service discovery, Ingress, and network policies in Kubernetes." }
{ "index": { "_index": "$Index", "_id": "doc16" } }
{ "title": "Service discovery and DNS", "content": "General service discovery and DNS concepts. Not Kubernetes-specific." }
{ "index": { "_index": "$Index", "_id": "doc17" } }
{ "title": "Ingress controller setup", "content": "NGINX Ingress for Kubernetes. TLS, routing, annotations." }
{ "index": { "_index": "$Index", "_id": "doc18" } }
{ "title": "CNI plugins overview", "content": "Kubernetes networking uses CNI. Calico, Cilium, Flannel comparison." }
{ "index": { "_index": "$Index", "_id": "doc19" } }
{ "title": "Mustache templates (general)", "content": "Mustache templating language basics. Not Elasticsearch." }
{ "index": { "_index": "$Index", "_id": "doc20" } }
{ "title": "Rank evaluation in IR", "content": "Information retrieval evaluation, judgments, qrels, graded relevance." }
{ "index": { "_index": "$Index", "_id": "doc21" } }
{ "title": "SAN-GEORGIO quickstart", "content": "Spelling variation SAN-GEORGIO. Quick manual notes and setup." }
{ "index": { "_index": "$Index", "_id": "doc22" } }
{ "title": "Dreams and sleep study", "content": "Scientific study about dreams during REM sleep. No ambition/goals." }
{ "index": { "_index": "$Index", "_id": "doc23" } }
{ "title": "Elasticsearch template rendering", "content": "Render templates and reuse query DSL. Practical examples." }
{ "index": { "_index": "$Index", "_id": "doc24" } }
{ "title": "Kubernetes DNS 문제 해결", "content": "kube-dns/CoreDNS troubleshooting guide. 서비스 디스커버리 이슈." }
"@


# NDJSON must end with newline
$bulk = $bulk.TrimEnd("`r","`n") + "`n"

$bulkArgs = @($curlCommon + @(
  "-X","POST","$EsUrl/_bulk",
  "-H","Content-Type: application/x-ndjson",
  "--data-binary","@-"
))
$bulk | & curl.exe @bulkArgs | Out-Host
if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): POST $EsUrl/_bulk" }

# [3/3] Refresh
Write-Host "[3/3] Refresh index"
$refArgs = @($curlCommon + @("-X","POST","$EsUrl/$Index/_refresh"))
& curl.exe @refArgs | Out-Host
if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE): POST $EsUrl/$Index/_refresh" }

Write-Host "Done. ES_URL=$EsUrl INDEX=$Index"
