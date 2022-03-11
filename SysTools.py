import cmath
import MathTools


def pole_calculator() -> None:
    print("Pole Calculator")

    for k in range(5):
        print(f'for K = {k} : {complex(-2 + cmath.sqrt(4 - 2 * k))}')
        print(f'for K = {k} : {complex(-2 - cmath.sqrt(4 - 2 * k))}')


def print_menu() -> None:
    print("\n"
          "0 Pole Calculator\n"
          "Press 'c' to close")


def main() -> None:
    choice = 'u'
    while choice != 'c':
        print_menu()
        choice = input()

        match choice:
            case '0':
                pole_calculator()
            case _:
                print("Input Error")


if __name__ == "__main__":
    main()
