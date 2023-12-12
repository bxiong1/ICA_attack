import pandas as pd
import csv
def load_ica_data(path):
    data = pd.read_csv(path, sep=',', encoding='utf-8')
    
    instructions = [ins for ins in data['instructions']]
    targets = [tar for tar in data['harmful_targets']]
    
    return instructions, targets

def load_instructions():
    file = open("/root/autodl-tmp/harmful_behaviors.csv", "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    instructions = []

    num = len(data)
    for i in range(num):
        if i==0:
            continue
        else:
            if len(data[i][0])==0 or len(data[i][1])==0:
                continue
            else:
                    instructions.append(data[i][0])
    return instructions


