import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TelegramBot():
    def __init__(self):
        super(TelegramBot, self).__init__()
        
        self.send_url = f"https://api.telegram.org/bot{os.environ['API_TOKEN']}/sendMessage"
        self.chat_id = os.environ['CHAT_ID']

    def send(self, message):
        json = {'chat_id': self.chat_id,
                'parse_mode': 'HTML',
                'text': message}
        requests.post(self.send_url, json=json)