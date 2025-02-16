from module5_mod import NumbersManager

def main():
    N = int(input("Enter N (positive integer): "))
    manager = NumbersManager()

    for i in range(N):
        number = int(input(f"Enter number {i+1}: "))
        manager.add_number(number)

    X = int(input("Enter X (integer): "))
    result = manager.find_number(X)
    print(result)

if __name__ == "__main__":
    main()