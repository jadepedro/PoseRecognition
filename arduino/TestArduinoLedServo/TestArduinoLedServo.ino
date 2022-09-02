#include <SerialLCD.h>
#include <SoftwareSerial.h> //this is a must
#include <math.h>
#include <Servo.h>

// LCD
// Init LCD display in PIN 11
SerialLCD slcd(11,12);
boolean backLight = false;

// Temperature sensor
const int TEMPERATURE_PIN = 0; // Temperature sensor is in PIN 0
int currentTemp = 0;   // value output temperature min
int sensorValue = 0;   // value read from the pot

// Led
const int LED_PIN = 10; // Led PIN 10

// Button
const int BUTTON_PIN = 8;
boolean buttonHigh = false;
int buttonValue = 0;
boolean fireSwitch = false;
boolean halfFire = false;

// Servo
const int SERVO_PIN = 2; // Servo motor PIN
Servo servo;
int servoPos = 0;

// Serial
int received = 0;

// Other
String cad = ".";

void setup()
{
  // Output pin setup
  pinMode(LED_PIN, OUTPUT);

  // Display setup
	slcd.begin();
	// Print a message to the LCD.
	slcd.print("Temp is:");
  slcd.noBacklight();
  slcd.setCursor(10,1);
  slcd.print("half=F");

  // Serial communications setup
  Serial.begin(115200);
  Serial.setTimeout(1);

  // Servo
  servo.attach(SERVO_PIN);
  for (servoPos = 0; servoPos <= 180; servoPos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    servo.write(servoPos);              // tell servo to go to position in variable 'pos'
    delay(15);                       // waits 15ms for the servo to reach the position
  }

}

void loop()
{
        //Read serial
        if (Serial.available())
        {
          // Read serial value
          received = Serial.readString().toInt();
          // Echo value with some operation
          Serial.print(received*10);
          updateServo(received);
        }
  
        //Read button and show status through led
        buttonValue = digitalRead(BUTTON_PIN);
        //Convierte a boolean
        buttonHigh = buttonValue == HIGH;
        if (!halfFire && buttonHigh)
        {
          halfFire = true;
          digitalWrite(LED_PIN, HIGH);
        }
        else if (halfFire && !buttonHigh)
        {
          halfFire = false;
          digitalWrite(LED_PIN, LOW);
          fireSwitch = true; 
        }
        // Event handler
        if (fireSwitch)
        {
          backLight = !backLight;
          fireSwitch = false;
          slcdPrintInfo(currentTemp, received, halfFire, buttonHigh, backLight);
        }
                     
      	currentTemp = readTemp();
       
}

float readTemp()
{
	const int B=3975; 
	double TEMP;
	int sensorValue = analogRead(TEMPERATURE_PIN);
	float Rsensor;
	Rsensor=(float)(1023-sensorValue)*10000/sensorValue;
	TEMP=1/(log(Rsensor/10000)/B+1/298.15)-273.15;
	return TEMP;
}

void updateServo(int servoValue)
{
  if (servoValue>=0 && servoValue<=180)
  {
        slcdPrintInfo(currentTemp, received, halfFire, buttonHigh, backLight);
        servo.write(servoValue);
  }
}

// Dont call this function in every iteration of the loop, as it interferes with servo control
void slcdPrintInfo(float currentTemp, int received, bool halfFire, bool buttonHigh, bool backLight)
{
  // Print char associated to received value
  slcd.setCursor(7,1);
  slcd.print(received);
  // Print received value as string in a different position
  char str[8];
  itoa( received, str, 10 ); //10 es la base numérica
  slcd.setCursor(10,0);
  slcd.print("   ");
  slcd.setCursor(10,0);
  slcd.print(str);

  if (halfFire)
  {
    slcd.setCursor(10,1);
    slcd.print("half=T");
  }
  else
  {
    slcd.setCursor(10,1);
    slcd.print("half=F");    
  }

  if (backLight)
  {
    slcd.backlight();
  }
  else
  {
    slcd.noBacklight();
  }

  // note: line 1 is the second row
  slcd.setCursor(0, 1);
  SLCDprintFloat( currentTemp ,1);
  slcd.print("C "); 
  
}

void SLCDprintFloat(double number, uint8_t digits) 
{ 
  // Handle negative numbers
  if (number < 0.0)
  {
     slcd.print('-');
     number = -number;
  }

  // Round correctly so that slcd.print(1.999, 2) prints as "2.00"
  double rounding = 0.5;
  for (uint8_t i=0; i<digits; ++i)
    rounding /= 10.0;
  
  number += rounding;

  // Extract the integer part of the number and print it
  unsigned long int_part = (unsigned long)number;
  float remainder = number - (float)int_part;
  slcd.print(int_part , DEC); // base DEC

  // Print the decimal point, but only if there are digits beyond
  if (digits > 0)
    slcd.print("."); 

  // Extract digits from the remainder one at a time
  while (digits-- > 0)
  {
    remainder *= 10.0;
    float toPrint = float(remainder);
    slcd.print(toPrint , DEC);//base DEC
    remainder -= toPrint; 
  } 
}
