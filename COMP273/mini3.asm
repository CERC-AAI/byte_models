.data
    prompt: .asciiz "Please input an integer value greater than or equal to 0: "
    zeroError: .asciiz "The value you entered is less than zero. This program only works with values greater than or equal to zero."
    inputMsg: .asciiz "Your input: "
    factorialMsg: .asciiz "\nThe factorial is: "
    repeatPrompt: .asciiz "\nWould you like to do this again (Y/N): "

.text
.globl main

main:
    # Initialize the loop
    loop_start:

    # Print the prompt to the user
    li $v0, 4
    la $a0, prompt
    syscall

    # Read integer from user input
    li $v0, 5
    syscall

    # Check if int is less than 0
    bltz $v0, lessThanZero

    # Store the input value for later use
    move $t1, $v0

    # Call factorial function
    move $a0, $v0
    jal factorial
    move $t0, $v0         # Store the factorial result in $t0

    # Print "Your input:"
    li $v0, 4
    la $a0, inputMsg
    syscall

    # Print the user input value
    li $v0, 1
    move $a0, $t1
    syscall

    # Print "The factorial is:"
    li $v0, 4
    la $a0, factorialMsg
    syscall

    # Print the factorial result
    li $v0, 1
    move $a0, $t0
    syscall

    # Prompt for repeat
    li $v0, 4
    la $a0, repeatPrompt
    syscall

    # Read single character input
    li $v0, 12
    syscall

    # Check if input is 'Y'
    li $t2, 'Y'
    beq $v0, $t2, loop_start

    # Exit the program
    li $v0, 10
    syscall

lessThanZero:
    li $v0, 4
    la $a0, zeroError
    syscall

    # Go back to the start of the loop
    j loop_start

# Factorial function definition follows...
factorial:
    addi $sp, $sp, -8    # Allocate space for 2 words
    sw $ra, 4($sp)       # Save return address
    sw $a0, 0($sp)       # Save the argument n

    # Base case: n = 0 or n = 1
    li $v0, 1            # Return 1
    blez $a0, endFactorial

    # Recursive case
    addi $a0, $a0, -1    # n = n - 1
    jal factorial        # Recursive call
    lw $t1, 0($sp)       # Load original n
    mul $v0, $v0, $t1    # n * factorial(n-1)

endFactorial:
    lw $ra, 4($sp)       # Restore return address
    addi $sp, $sp, 8     # Deallocate stack space
    jr $ra               # Return to caller

