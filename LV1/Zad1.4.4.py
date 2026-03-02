fileHandler = open("LV1/song.txt")

dict = {}

for line in fileHandler:
    line = line.strip()
    for word in line.split():
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1

onlyOneCounter = 0

for key, value in dict.items():
    if(dict[key] == 1): onlyOneCounter += 1

print("Broj rijeci koje se pojavljuju samo jednom: " + str(onlyOneCounter))