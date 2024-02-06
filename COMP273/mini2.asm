.data
buffer: .space 31
buffer2: .space 31
userInt: .word 0
inputPrompt: .asciiz "Input a string 30 characters or less: "
numberPrompt: .asciiz "\nInput an integer greater than 0: "
errorMsg: .asciiz "\nNo input. Run again."
wrongInputMsg: .asciiz "\nWrong input. Run again."
shiftedMsg: .asciiz "\nShifted string = ["
closeBracket: .asciiz "]"

.text
.globl main
#.globl emptyInput

main:
    # print string input
    la $a0, inputPrompt
    li $v0, 4
    syscall

    # record string input
    la $a0, buffer
    li $a1, 31
    li $v0, 8
    syscall


    # Check if the string is "empty" (only newline and null character)
    lb $t0, 0($a0)
    li $t1, 10
    beq $t0, $t1, emptyInput


    # print int
    la $a0, numberPrompt
    li $v0, 4
    syscall

    # record int input
    li $v0, 5
    syscall
    sw $v0, userInt

    # check if int is leq 0
    lw $t0, userInt
    blez $t0, lessThanOrEqualToZero

    # shifting logic
    la $a0, buffer
    la $a1, buffer2
    lw $t0, userInt

    # copying logic
    add $t3, $a0, $t0  # $t3 points to the start of the substring in buffer
    copy_loop_1:
        lb $t1, 0($t3)
        beqz $t1, end_copy_1  # Break if null
        sb $t1, 0($a1)
        addiu $t3, $t3, 1
        addiu $a1, $a1, 1
        j copy_loop_1

    end_copy_1:
    # $a1 points to where copying needs to continue

    move $t3, $a0

    # Copy the first n bytes to buffer2
    li $t2, 0
    copy_loop_2:
        bge $t2, $t0, end_copy_2  # Break if n bytes copied
        lb $t1, 0($t3)
        sb $t1, 0($a1)
        addiu $t2, $t2, 1
        addiu $t3, $t3, 1
        addiu $a1, $a1, 1
        j copy_loop_2

    end_copy_2:
    # Null-terminate buffer2
    sb $zero, 0($a1)


    # Print Shifted string = [
    la $a0, shiftedMsg
    li $v0, 4
    syscall

    # Print buffer2
    la $a0, buffer2
    li $v0, 4
    syscall

    # Print "]"
    la $a0, closeBracket
    li $v0, 4
    syscall

    j end_program


emptyInput:
    # Handle empty input
    la $a0, errorMsg
    li $v0, 4
    syscall

    j end_program

lessThanOrEqualToZero:
    # Handle empty input
    move $a0, $zero
    la $a0, wrongInputMsg
    li $v0, 4
    syscall

    j end_program

end_program:
    # System call for end program
    li $v0, 10
    syscall
