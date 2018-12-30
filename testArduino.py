import serial 

#serial.write(serialcmd.encode())
arduinoData = serial.Serial('/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_754333137393517061D0-if00', 9600)

arduinoData.write('0'.encode())

arduinoData.write('1'.encode())

## https://stackoverflow.com/questions/23669855/linux-pyserial-could-not-open-port-dev-ttyama0-no-such-file-or-directory
## https://www.youtube.com/watch?v=OleCp_TAXC8
