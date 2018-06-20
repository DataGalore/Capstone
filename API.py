# # API wrangling
# import requests
# import json
# # params = {"api_key": "tSKqprcg7k-S","format": "csv"}
# response = requests.get("http://api.open-notify.org/astros.json")
# print(response.text)

# with open('Output.csv', 'w+') as f:
#     f.write(response.text)
import regex as re
Months = ["Aug2001", "Nov2007"]
string="This is a string that contains #134534 and other things"
match= [re.findall(r'\d+',m) for m in Months];

print (match)
# [int(s) for s in str.split() if s.isdigit()]
# [23, 11, 2]