userInput = ""

numbers = []    

while True:
    userInput = input("Unesi broj (Done za kraj): ")
    if userInput == "Done": 
        break
    try:
        numbers.append(float(userInput))
    except ValueError:
        print("Unos mora biti broj ili 'Done' za kraj")

if numbers:
    print("Broj elemenata:", len(numbers))
    print("Prosjek:", sum(numbers) / len(numbers))
    print("Najveći broj:", max(numbers))
    print("Najmanji broj:", min(numbers))
else:
    print("Nema unesenih brojeva")