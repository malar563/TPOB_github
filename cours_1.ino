void setup() {
  // put your setup code here, to run once:
  int delayON = 1000;
  int delayOFF = 1000;

  Serial.begin(9600);

  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(9, HIGH);  // turn the LED on (HIGH is the voltage level)
  digitalWrite(10, HIGH);
  delay(delayON);                      // wait for a second
  digitalWrite(9, LOW);   // turn the LED off by making the voltage LOW
  digitalWrite(10, HIGH);   // turn the LED off by making the voltage LOW
  delay(delayOFF);

}
