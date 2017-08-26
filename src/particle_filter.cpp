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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	default_random_engine gen;
	double std_x, std_y, std_theta;
	num_particles = 50;
	
	// set standard deviations
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// create normal distributions
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// initilize all particles
	for(int i=0; i<num_particles; i++)
	{
		// sample from normal distributions
		double sample_x = dist_x(gen);
		double sample_y = dist_y(gen);
		double sample_theta = dist_theta(gen);

		// create particles
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;
	
		particles.push_back(particle);
		weights.push_back(1);
	}
	is_initialized = true;
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	double std_x, std_y, std_theta;
	
	// set standard deviations
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for(int i=0; i<num_particles; i++)
	{
		// extract x, y, theta
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// create normal distributions
		normal_distribution<double> dist_x(x, std_x);
		normal_distribution<double> dist_y(y, std_y);
		normal_distribution<double> dist_theta(theta, std_theta);

		// sample from normal distributions
		double sample_x = dist_x(gen);
		double sample_y = dist_y(gen);
		double sample_theta = dist_theta(gen);

		// create particles
		if (fabs(yaw_rate) > 0.001)
		{
			particles[i].x = sample_x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y = sample_y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta = sample_theta + yaw_rate * delta_t;
		}
		else
		{
			particles[i].x = sample_x + velocity * cos(theta);
			particles[i].y = sample_y + velocity * sin(theta);
			particles[i].theta = sample_theta;
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for(int i=0; i<predicted.size(); i++)
	{
		double closest_distance = 100000;
		LandmarkObs* closest_observation = NULL;

		for(int j=0; j<observations.size(); j++)
		{
			double this_distance = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
			if(this_distance < closest_distance)
			{
				closest_distance = this_distance;
				closest_observation = &observations[j];
				observations[j].id = predicted[i].id;
			}
		}
	}
}

double multivariate_gaussian(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) 
	{
	return exp(-(
		  (pow(x - mu_x, 2) / (2 * pow(sig_x, 2)) +
		   pow(y - mu_y, 2) / (2 * pow(sig_y, 2))
		  ))) / (2 * M_PI * sig_x * sig_y);
	}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		vector<LandmarkObs> transformed_observations;
	
		for (int j = 0; j < observations.size(); j++) {
		  LandmarkObs obs = observations[j];
		  LandmarkObs transformed;
	
		  transformed.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
		  transformed.y = obs.y * cos(p.theta) + obs.x * sin(p.theta) + p.y;
		  transformed.id = 0;
	
		  transformed_observations.push_back(transformed);
		}
	
		vector<LandmarkObs> predicted;
	
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
		  Map::single_landmark_s l = map_landmarks.landmark_list[j];
		  double distance = dist(p.x, p.y, l.x_f, l.y_f);
	
		  if (distance < sensor_range) {
			LandmarkObs obs;
			obs.x = l.x_f;
			obs.y = l.y_f;
			obs.id = l.id_i;
			predicted.push_back(obs);
		  }
		}
	
		if (predicted.size() > 0) {
		  dataAssociation(predicted, transformed_observations);
	
		  p.weight = 1;
	
		  for (int j = 0; j < transformed_observations.size(); j++) {
			LandmarkObs obs = transformed_observations[j];
			Map::single_landmark_s l = map_landmarks.landmark_list[obs.id - 1];
			double weight = multivariate_gaussian(
				l.x_f, l.y_f, obs.x, obs.y, std_landmark[0], std_landmark[1]);
	
			p.weight *= weight;
		  }
		} else {
		  p.weight = 0;
		}
	
		weights[i] = p.weight;
		particles[i].weight = p.weight;
	  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double beta = 0.0;
	random_device rd;
	mt19937 genindex(rd());
	uniform_int_distribution<int> dis(0, num_particles - 1);
	int index = dis(genindex);
  	vector<Particle> resampled_particles;
  	double max_weight = 0.0;

	resampled_particles.reserve(num_particles);
  	weights.clear();
	weights.reserve(num_particles);

	for (Particle particle : particles) {
	  if (particle.weight > max_weight) {
			max_weight = particle.weight;
	  }

	  weights.push_back(particle.weight);
	}

  	mt19937 gen(rd());
	uniform_real_distribution<double> dis_real(0, 2.0 * max_weight);

	for (Particle particle : particles) {
	  beta += dis_real(gen);
  
	  while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
	  }
  
	  resampled_particles.push_back(particles[index]);
	}

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
