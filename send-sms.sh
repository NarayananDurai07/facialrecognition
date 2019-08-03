curl -X "POST" "https://rest.nexmo.com/sms/json" \
  -d "from=AcmeInc" \
  -d "text=A text message sent using the Nexmo SMS API" \
  -d "to=+919663317875" \
  -d "api_key=6116" \
  -d "api_secret=8197"

sleep 2s

echo $1