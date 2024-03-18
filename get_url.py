import requests
import json

url = 'https://account.divvybikes.com/bikesharefe-gql'
headers = {'content-type': 'application/json'}
data = {'query': 'query { __typename }'}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.text)
