#import chess #https://github.com/niklasf/python-chess
import chess.engine
import serial
import time
from fresher10 import takePicture, partitionPicture, findMove

ser=serial.Serial('COM3', 9600, timeout=None)
engine = chess.engine.SimpleEngine.popen_uci("C:\\Users\Eric\Desktop\Recycle Bin\Work\Year 5\FYP\Chess Engine\stockfish-10-win\Windows\stockfish_10_x64")
Difficulty = 4
engine.configure({"Skill Level": Difficulty})
#==========================================
#Global Variables
#==========================================
QuitFlag=0
board = chess.Board()
CurrentTurn="Black"
#==========================================
#Main
#==========================================
savePicPathR = takePicture('boop', noFrames=5)
moveCode, sqColors, sqIDataArchive, sqOccupancyArchive = findMove(savePicPathR)

while (QuitFlag==0):
    print(board)
    print("===============================")
    
    #Check if win conditions are met
    if (board.is_checkmate()):
        QuitFlag=1
        print("Winner: " + CurrentTurn)
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('4'.encode('utf-8'))
        break
    elif (board.is_stalemate()):
        QuitFlag=1
        print("Stalemate")
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('4'.encode('utf-8'))
        break
    elif (board.is_seventyfive_moves()):
        QuitFlag=1
        print("Draw by 75 moves rule")
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('0'.encode('utf-8'))
        ser.write('4'.encode('utf-8'))
        break
    elif (board.is_check()):
        if (CurrentTurn=="White"):
            print("Black is checked!")
        else:
            print("White is checked!")
        
    #Change Turn
    if (CurrentTurn=="White"):
        CurrentTurn="Black"
    else:
        CurrentTurn="White"

    #Let computer take the move
    if (CurrentTurn=="Black"):
        result = engine.play(board, chess.engine.Limit(time=0.500))
        print('Computer Moves: ' + str(result.move))
        SerialFrom=str(result.move.from_square).zfill(2)
            
        SerialTo=str(result.move.to_square).zfill(2)
        ser.write(SerialFrom.encode('utf-8')) #Source square
        ser.write(SerialTo.encode('utf-8')) #To square
            
        #Check anything other than basic move
        if result.move.promotion==5:
            ser.write('1'.encode('utf-8'))
        elif (board.is_castling(result.move)):
            ser.write('2'.encode('utf-8'))
        elif  (board.is_capture(result.move)):
            ser.write('3'.encode('utf-8'))
        else:
            ser.write('0'.encode('utf-8'))
        board.push(result.move)
    
    #Let player take the move
    if (CurrentTurn=="White"):
        ser.reset_input_buffer()
        time.sleep(1)
        Data = ser.read(2)
        print(Data)

        savePicPathR = takePicture('boop', noFrames=5)
        moveCode, sqColors, sqIDataArchive, sqOccupancyArchive = findMove(savePicPathR,
                                                                          sqColors=sqColors,
                                                                          sqIDataArchive=sqIDataArchive,
                                                                          sqOccupancyArchive=sqOccupancyArchive)
        if (Data == b'\x01\x01'): #Reset
            board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
            CurrentTurn = "Black"
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('5'.encode('utf-8'))
        elif (Data == b'\x00\x01'): #Undo
            board.pop() #Pop twice (once for computer move, other for previous move)
            board.pop()
            CurrentTurn = "Black"
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('0'.encode('utf-8'))
            ser.write('5'.encode('utf-8'))
        else:
            PlayerInput=moveCode
            print(moveCode)
            PlayerInput=PlayerInput.lower()
            while (PlayerInput[0:2] not in chess.SQUARE_NAMES) or (PlayerInput[2:] not in chess.SQUARE_NAMES):
                PlayerInput=input("Invalid Moveset. Enter [a-h][1-8][a-h][1-8]")
            PlayerMove=chess.Move.from_uci(PlayerInput)
            if PlayerMove not in board.legal_moves:
                while PlayerMove not in board.legal_moves:
                    PlayerInput=input("Invalid Move. Enter a valid move: ")
                    PlayerMove=chess.Move.from_uci(PlayerInput)
            board.push(PlayerMove)
                
engine.quit()
ser.close()
