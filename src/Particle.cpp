#include "Particle.h"
#include <stdexcept>

//Constructor
Particle::Particle(double m, double x0, double y0) 
    : mass(m), x(x0), y(y0), vx(0), vy(0), ax(0), ay(0) 
{
    if (m < 0) {
        throw std::invalid_argument("La masa no puede ser negativa");
    }
}

//Resetea la aceleracion a cero
void Particle::resetAcceleration() {
    ax = 0;
    ay = 0;
}

void Particle::setAcceleration(double ax_, double ay_) {
    ax = ax_;
    ay = ay_;
}

void Particle::addAcceleration(double dax, double day) {
    ax += dax;
    ay += day;
}

void Particle::kick(double dt) {
    vx += ax * dt;
    vy += ay * dt;
}

void Particle::drift(double dt) {
    x += vx * dt;
    y += vy * dt;
}

void Particle::setVx(double vx_) {
     vx = vx_; 
    }
void Particle::setVy(double vy_) {
     vy = vy_; 
    }

//Getters 

double Particle::getMass() const {
    return mass;
}

double Particle::getX() const {
    return x;
}

double Particle::getY() const {
    return y;
}

double Particle::getVx() const {
    return vx;
}

double Particle::getVy() const {
    return vy;
}

double Particle::getAx() const {
    return ax;
}

double Particle::getAy() const {
    return ay;
}

