//Conjunto de partículas, G, epsilon; computeAccelerations*

class NBodySystem {
private:
    std::vector<Particle> bodies;
    double G_const;
    double softening_eps;
public:
    NBodySystem(double G, double epsilon);
    void addParticle(const Particle& p);
    void zeroAccelerations();

// Sobrecarga: c ́alculo de aceleraciones con distintos schedules / collapse
    void computeAccelerations();
    void computeAccelerations(int schedule_type);
    void computeAccelerations(int schedule_type, int chunk_size);
    void computeAccelerationsCollapse(); // p.ej. collapse(2) en i,j
    const std::vector<Particle>& getBodies() const;
    int getCount() const;
};