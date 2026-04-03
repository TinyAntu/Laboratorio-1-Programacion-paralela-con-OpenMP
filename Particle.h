// Masa, posicion, velocidad, aceleraci ́on (2D)

class Particle {
private:
    double mass;
    double x, y, vx, vy, ax, ay; //Posicion, velocidad, aceleracion
public:
    Particle(double m, double x0, double y0);
    void setAcceleration(double ax_, double ay_); 
    void addAcceleration(double dax, double day);
    void kick(double dt); // v += a*dt
    void drift(double dt); // r += v*dt
    double getMass() const;
    // getters/setters de estado...
};