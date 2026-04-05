// Masa, posicion, velocidad, aceleraci ́on (2D)
//Particle: estado 2D del cuerpo y acceso controlado
#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
private:
    double mass;
    double x, y, vx, vy, ax, ay; //Posicion, velocidad, aceleracion
public:
    Particle(double m, double x0, double y0);
    void resetAcceleration();
    void setAcceleration(double ax_, double ay_); 
    void addAcceleration(double dax, double day);
    void kick(double dt); // v += a*dt
    void drift(double dt); // r += v*dt

    // getters/setters de estado...
    double getMass() const;

    double getX() const;
    double getY() const;        

    double getVx() const;
    double getVy() const;

    double getAx() const;
    double getAy() const;
};

#endif // PARTICLE_H