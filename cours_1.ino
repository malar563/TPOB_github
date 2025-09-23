int delayRouge = 1000;
int delayIR = 1000;

const int delRouge = 9;  // Pin pour DEL rouge
const int delIR = 10;     // Pin pour DEL IR
const int pinRouge = A0;
const int pinIR = A1;

const int nLoop = 500;   // Nombre de loops
int dataRouge[nLoop];
int dataIR[nLoop];



void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);

  pinMode(delRouge, OUTPUT);
  pinMode(delIR, OUTPUT);


    // Lire plus souvent le analog input? À chaque fois qu'on change la DEL qui s'allume?
    // Mettre le analogRead() au bon endroit. 
    // Mettre allumage à chaque 0.5s et mettre petit délai entre?
    // Comment mettre en csv

  for (int i = 0; i < nLoop; i++) {
    // Allume DEL rouge
    digitalWrite(delRouge, HIGH);
    digitalWrite(delIR, LOW);
    delay(delayRouge); // Temps d'illumination
    dataRouge[i] = analogRead(pinRouge); // Lire plus souvent?

    // Allume DEL IR
    digitalWrite(delRouge, LOW);
    digitalWrite(delIR, HIGH);
    delay(delayIR); // Temps d'illumination
    dataIR[i] = analogRead(pinIR); // Mettre dans une autre liste?
  }

  // Print les données
  Serial.println(Rouge, IR);
  for (int i = 0; i < nLoop; i++) {
    Serial.print(dataRouge[i]);
    Serial.print(",");
    Serial.println(dataIR[i]);
  }
}


void loop() {

}
