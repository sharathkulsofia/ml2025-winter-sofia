def main():
    # Ask the user for a positive integer N
    N = int(input("Enter a positive integer N: "))
    
    # Initialize an empty list to store the numbers
    numbers = []
    
    # Read N numbers from the user, one by one
    for i in range(N):
        num = int(input(f"Enter number {i+1}: "))
        numbers.append(num)
    
    # Ask for the integer X to search in the list
    X = int(input("Enter the integer X: "))
    
    # Try to find the first occurrence of X (1-indexed)
    try:
        index = numbers.index(X)
        print(index + 1)  # convert from 0-indexed to 1-indexed
    except ValueError:
        # X was not found in the list
        print("-1")

if __name__ == "__main__":
    main()