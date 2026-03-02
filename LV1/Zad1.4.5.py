fileHandler = open("LV1/SMSSpamCollection.txt")

hamWordCounter = 0
spamWordCounter = 0
spamEndsWithQuestionMarkCounter = 0

for line in fileHandler:
    line = line.strip()
    if line.startswith("ham"):
        for word in line.split():
            hamWordCounter += 1
    elif line.startswith("spam"):
        if line.endswith("?"): spamEndsWithQuestionMarkCounter += 1
        for word in line.split():
            spamWordCounter += 1

print("Prosjecan broj rijeci u ham porukama: " + str(hamWordCounter/5574))
print("Prosjecan Broj rijeci u spam porukama: " + str(spamWordCounter/5574))
print("Broj spam poruka koje zavrsavaju sa '?': " + str(spamEndsWithQuestionMarkCounter))
