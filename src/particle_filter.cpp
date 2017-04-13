/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

#define PI 3.14159265358979323846  /* pi */
# define MIN_YAW_RATE 0.00001 // Check for safe division

void ParticleFilter::init(const double x, const double y, const double theta, const double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 5;

    std::default_random_engine gen;

    // Create a normal (Gaussian) distribution for x
    std::normal_distribution<double> dist_x(x, std[0]);

    // Create normal distributions for y and psi
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_psi(theta, std[2]);

    // Clear all particles and weights
    particles.clear();
    weights.clear();
    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle new_particle;

        // Sample  and from these normal distributions like this: 
        //	 sample_x = dist_x(gen);
        //	 where "gen" is the random engine initialized earlier.

        new_particle.x      = dist_x(gen);
        new_particle.y      = dist_y(gen);
        new_particle.theta  = dist_psi(gen);	 
        new_particle.weight = 1.0;
        new_particle.id     = i;

        // Add this new particle to set of particles.
        particles.push_back(new_particle);
        weights.push_back(1.0F);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(const double delta_t, const double std_pos[], const double velocity, const double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;

    // Create normal distributions for x, y and psi
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    // Pre-calculate pre-factor for motion step:
    const bool yawRateNotZero = std::fabs(yaw_rate) > MIN_YAW_RATE;
    const double fac = yawRateNotZero? velocity/yaw_rate : velocity;
    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle& current_particle = particles[i];

        if (yawRateNotZero)
        {
            const double theta_new   = current_particle.theta + yaw_rate*delta_t;
            const double theta_old   = current_particle.theta;
            const double sinThetaNew = std::sin(theta_new);
            const double sinThetaOld = std::sin(theta_old);
            const double cosThetaNew = std::cos(theta_new);
            const double cosThetaOld = std::cos(theta_old);

            current_particle.x      += fac*(sinThetaNew-sinThetaOld);
            current_particle.y      += fac*(cosThetaOld-cosThetaNew);
            current_particle.theta   = theta_new;
        }
        else
        {
            const double sinTheta = std::sin(current_particle.theta);
            const double cosTheta = std::cos(current_particle.theta);

            current_particle.x     += fac*cosTheta*delta_t;
            current_particle.y     += fac*sinTheta*delta_t;
        }

        // Add noise:
        const double noise_x     = dist_x(gen);
        const double noise_y     = dist_y(gen);
        const double noise_theta = dist_theta(gen);

        current_particle.x      += dist_x(gen);
        current_particle.y      += dist_y(gen);
        current_particle.theta  += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& /*predicted*/, std::vector<LandmarkObs>& /*observations*/) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}

void ParticleFilter::updateWeights(const double sensor_range, const double std_landmark[], 
		const std::vector<LandmarkObs>& observations, const Map& map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    const double sensor_range_squared = sensor_range*sensor_range;

    for (unsigned int i = 0; i < num_particles; ++i)
    {
        Particle& cur_particle        = particles[i];
        const double cosThetaParticle = std::cos(cur_particle.theta);
        const double sinThetaParticle = std::sin(cur_particle.theta);
        const double xParticle        = cur_particle.x;
        const double yParticle        = cur_particle.y;

        std::vector<LandmarkObs> meas_map;
        for (unsigned int j = 0; j < observations.size(); ++j)
        {
            const LandmarkObs& cur_meas_veh = observations[j];
            LandmarkObs cur_meas_map;

            //Transform from vehicle to map
            cur_meas_map.x = xParticle  + cur_meas_veh.x*cosThetaParticle - cur_meas_veh.y*sinThetaParticle;
            cur_meas_map.y = yParticle  + cur_meas_veh.x*sinThetaParticle + cur_meas_veh.y*cosThetaParticle;
            meas_map.push_back(cur_meas_map);
        }

        cur_particle.weight = 1.0;

        for(unsigned int j = 0; j < meas_map.size(); ++j)
        {
            const LandmarkObs& cur_mea_map = meas_map[j];
            double min_dist_squared = sensor_range_squared;
            const Map::single_landmark_s* closestLm = NULL;

            for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) 
            {
                const Map::single_landmark_s& cur_lm = map_landmarks.landmark_list[k];
                const double x_lm = cur_lm.x_f;
                const double y_lm = cur_lm.y_f;

                const double cur_dist_squared = distSquared(cur_mea_map.x, cur_mea_map.y, x_lm, y_lm);
                if(cur_dist_squared < min_dist_squared)
                {
                    min_dist_squared = cur_dist_squared;
                    closestLm = &cur_lm;
                }
            }

            if(closestLm)
            {
                const double x_meas   = cur_mea_map.x;
                const double y_meas   = cur_mea_map.y;
                const double x_mu     = closestLm->x_f;
                const double y_mu     = closestLm->y_f;
                const double measProb = 1/(2.0*PI*std_landmark[0]*std_landmark[1])*std::exp(-(std::pow(x_meas-x_mu,2.0)/(2.0*pow(std_landmark[0],2.0))+pow(y_meas-y_mu,2.0)/(2.0*pow(std_landmark[1],2.0))));
                cur_particle.weight   *= measProb;
            }
        }
        weights[i] = cur_particle.weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
#if 0
    std::default_random_engine gen;
    // Create uniform distribution from 0 ... 1
    std::discrete_distribution<int> dist(weights.cbegin(), weights.cend());

    std::vector<Particle> new_particles;
    for (unsigned int i = 0; i < num_particles; ++i)
    {
        new_particles.push_back(particles[dist(gen)]);
    }
    particles = new_particles;
#endif

    std::default_random_engine gen;
    std::uniform_real_distribution<> dist(0, 1);
    std::vector<Particle> new_particles;
    unsigned int index = std::rand()%num_particles;
    double beta = 0.0;
    const double max_weight = *std::max_element(weights.cbegin(), weights.cend());
    for (unsigned int i = 0; i < num_particles; ++i)
    {
        beta += dist(gen)*2.0*max_weight;
        while (beta > weights[index])
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    } 
    particles = new_particles;  
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (unsigned int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
