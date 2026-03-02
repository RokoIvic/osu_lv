try: 
    grade = float(input("Upisi ocjenu: "))

    if(grade > 1.0 or grade < 0.0): raise Exception("Ocjena mora biti između 1.0 i 0.0")

    if(grade >= 0.9): print("A")
    elif(grade >= 0.8): print("B")
    elif(grade >= 0.7): print("C")
    elif(grade >= 0.6): print("D")
    elif(grade < 0.6): print("F")
except ValueError:
    print("Ocjena mora biti broj")

