import os
import telegram
import numpy as np
from datetime import datetime
from pytz import timezone


# Alaram settings
RECORD_T = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M')[2:]
TOKEN = '6740952693:AAFOUwNFVu2O3Bpf7nlKwIlDzyNaarN7Fl8'
CHAT_ID = '5110804803'
MSG = f"Lossmap was generated!\nPlease check and zoom it!"

bot = telegram.Bot(TOKEN)

# Make directories
def canonical_name(record_t=RECORD_T):
    return f"ResNet_{record_t}"

def send_alaram(msg=MSG):
    return bot.sendMessage(chat_id=CHAT_ID, text=msg)

output_path = f"./output/{canonical_name()}"
os.makedirs(output_path + "/train/accuracy/fd", exist_ok=True)
os.makedirs(output_path + "/train/accuracy/border", exist_ok=True)
os.makedirs(output_path + "/train/loss/fd", exist_ok=True)
os.makedirs(output_path + "/train/loss/border", exist_ok=True)
os.makedirs(output_path + "/test/accuracy/fd", exist_ok=True)
os.makedirs(output_path + "/test/accuracy/border", exist_ok=True)
os.makedirs(output_path + "/test/loss/fd", exist_ok=True)
os.makedirs(output_path + "/test/loss/border", exist_ok=True)


def save_outputs(mode='train', *args):
    np.save(output_path + '/' + mode + '.npy')
