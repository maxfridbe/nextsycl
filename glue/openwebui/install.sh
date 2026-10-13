#!/bin/sh
# install.sh [OPEN_WEBUI_URL] [FRONT_DOOR_URL]: puts nextsycl_speed.py into Open WebUI as a global filter function
# (pp / tg speeds above each answer, live), its valves set to the front door and its API token. Open WebUI's admin
# token: OWUI_TOKEN, else a sign-in (works as is with WEBUI_AUTH=False). Run it again to update the function.
set -eu
owui=${1:-http://127.0.0.1:8080}
door=${2:-http://127.0.0.1:8000}
here=$(dirname "$0")
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
tok=${OWUI_TOKEN:-$(curl -sf -X POST "$owui/api/v1/auths/signin" -H 'Content-Type: application/json' -d '{"email":"","password":""}' | json 'd["token"]')}
door_tok=${NS_API_TOKEN:-$(curl -sf "$door/api/spec" | json 'd["token"]')}
body=$(python3 -c '
import json,sys
src=open(sys.argv[1]).read()
print(json.dumps({"id":"nextsycl_speed","name":"nextsycl speed","content":src,
                  "meta":{"description":"pp / tg speeds above each answer, live, as the nextsycl chat server reports them"}}))' "$here/nextsycl_speed.py")
auth="Authorization: Bearer $tok"
if curl -sf -H "$auth" "$owui/api/v1/functions/id/nextsycl_speed" >/dev/null; then
  curl -sf -X POST -H "$auth" -H 'Content-Type: application/json' "$owui/api/v1/functions/id/nextsycl_speed/update" -d "$body" >/dev/null
  echo "updated nextsycl_speed"
else
  curl -sf -X POST -H "$auth" -H 'Content-Type: application/json' "$owui/api/v1/functions/create" -d "$body" >/dev/null
  echo "created nextsycl_speed"
fi
state=$(curl -sf -H "$auth" "$owui/api/v1/functions/id/nextsycl_speed")
[ "$(echo "$state" | json 'd["is_active"]')" = True ] || curl -sf -X POST -H "$auth" "$owui/api/v1/functions/id/nextsycl_speed/toggle" >/dev/null
[ "$(echo "$state" | json 'd["is_global"]')" = True ] || curl -sf -X POST -H "$auth" "$owui/api/v1/functions/id/nextsycl_speed/toggle/global" >/dev/null
curl -sf -X POST -H "$auth" -H 'Content-Type: application/json' "$owui/api/v1/functions/id/nextsycl_speed/valves/update" \
  -d "{\"url\": \"$door\", \"token\": \"$door_tok\", \"interval\": 0.5, \"priority\": 0}" >/dev/null
echo "active and global; valves: $door"
