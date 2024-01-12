# Implementation of Stack
class Stack:
    def __init__(self):
        # initialize a list
        self.items = []

    def empty(self):
        # Check if there are elements in the list
        if len(self.items) == 0:
            return True
        else:
            return False

    def push(self, value):
        # insert the values at the first position of the list
        self.items.insert(0, value)

    def pop(self):
        # if the list is not empty it removes first element of the list
        if not self.empty():
            return self.items.pop(0)
        else:
            return "Not Enough Elements to Remove"


def balanced(str1):
    stack = Stack()
    error = False
    # iterate the given string
    for elements in str1:
        # check if the current symbol is left (, {, [ and if it is we push the element into the stack
        if elements == '{' or elements == '(' or elements == '[':
            stack.push(elements)
        # if the current symbol is right ), }, ] and the stack is empty it means the string is not balanced
        else:
            if stack.empty():
                error = True
                break

            lcheck = elements
            str2 = stack.pop()
            # pop one element and check if it matches with right ], }, ), if it does not match the string
            # is not balanced
            if str2 == '[' and not lcheck == ']':
                error = True
                break
            if str2 == '{' and not lcheck == '}':
                error = True
                break
            if str2 == '(' and not lcheck == ')':
                error = True
                break
    # if the stack is empty without any mismatched doubles the string is balanced
    if stack.empty() and not error:
        return True
    else:
        return False


if __name__ == '__main__':
    print("Type a String with parentheses to check if it is balanced")
    string = str(input())
    if balanced(string):
        print("Balanced")
    else:
        print("Not Balanced")
