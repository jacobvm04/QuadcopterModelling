import numpy as np
from numpy import sin, cos, tan

def rotate(vector, rotation):
	phi = -rotation[0]
	theta = rotation[2]
	psi = rotation[1]

	rotationMatrixZ = np.array(
		[
			[cos(phi), -sin(phi), 0],
			[sin(phi), cos(phi), 0],
			[0, 0, 1],
		]
	)
	rotationMatrixY = np.array(
		[
			[cos(theta), 0, sin(theta)],
			[0, 1, 0],
			[-sin(theta), 0, cos(theta)],
		]
	)
	rotationMatrixX = np.array(
		[
			[1, 0, 0],
			[0, cos(psi), -sin(psi)],
			[0, sin(psi), cos(psi)],
		]
	)

	rotationMatrix = np.matmul(np.matmul(rotationMatrixZ, rotationMatrixY), rotationMatrixX)
	return rotationMatrix.dot(vector)

def motorThrustVector(angularVelocitySquared, thrustCoefficient, numMotors):
	return np.array([
		0,
		0,
		numMotors * thrustCoefficient * angularVelocitySquared.sum()
	])

def gravityVector(mass, gravityConstant):
	return np.array([
		0,
		0,
		-mass * gravityConstant
	])

def dragVector(velocity, dragCoefficient):
	return np.array([
		-dragCoefficient * velocity[0],
		-dragCoefficient * velocity[1],
		-dragCoefficient * velocity[2]
	])

def eulerRateVelocity(eulerAngles, angularVelocity):
	phi = eulerAngles[0]
	theta = eulerAngles[1]
	psi = eulerAngles[2]
	p = angularVelocity[0]
	q = angularVelocity[1]
	r = angularVelocity[2]

	return np.array([
		(q * sin(psi) + p * cos(psi)) * (1 / cos(theta)),
		q * cos(psi) - p * sin(psi),
		r + (q * sin(psi) + p * cos(psi)) * tan(theta)
	])

def torqueVector(angularVelocitySquared, distanceToMotor, thrustCoefficient, dragCoefficient):
	return np.array([
		distanceToMotor * thrustCoefficient * ((angularVelocitySquared[0] + angularVelocitySquared[1]) - (angularVelocitySquared[2] + angularVelocitySquared[3])), # y, pitch
		distanceToMotor * thrustCoefficient * ((angularVelocitySquared[0] + angularVelocitySquared[3]) - (angularVelocitySquared[1] + angularVelocitySquared[2])), # x, roll
		dragCoefficient * (angularVelocitySquared[0] - angularVelocitySquared[1] + angularVelocitySquared[2] - angularVelocitySquared[3]) # z, yaw
	])

class PIDController:
	def __init__(self, kp, ki, kd):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.integral = 0
		self.lastError = 0

	def update(self, error):
		self.integral = self.integral + error
		derivative = error - self.lastError
		self.lastError = error
		return self.kp * error + self.ki * self.integral + self.kd * derivative

class Quadcopter:
    def __init__(self, mass, thrustCoefficient, dragCoefficient, gravity, inertiaTensor):
        self.mass = mass
        self.massInverse = 1/mass
        self.thrustCoefficient = thrustCoefficient
        self.dragCoefficient = dragCoefficient
        self.gravity = gravity
        self.inertiaTensor = inertiaTensor

        self.x = np.array([0, 0, 0])
        self.xdot = np.array([0, 0, 0])
        self.theta = np.array([0, 0, 0])
        self.thetadot = np.array([0, 0, 0])
        self.omega = np.array([0, 0, 0])
        self.motorAngularVelocity = np.array([9000, 9000, 9000, 9000])
    
    def update(self, dt):
        xdotdot = self.massInverse * rotate(motorThrustVector(self.motorAngularVelocity, self.thrustCoefficient, 4), self.theta) + self.massInverse * gravityVector(self.mass, self.gravity) + dragVector(self.xdot, self.dragCoefficient)
        self.xdot = self.xdot + xdotdot * dt
        self.x = self.x + self.xdot * dt

        torques = torqueVector(self.motorAngularVelocity, 0.05, self.thrustCoefficient, 0.00001)
        omegatemp = np.array([
			self.omega[1],
			self.omega[2],
			self.omega[0]
		])
        omegadot = np.dot(np.linalg.inv(self.inertiaTensor), torques - np.cross(omegatemp, np.dot(self.inertiaTensor, omegatemp)))
        omegadot = np.array([
			omegadot[2], # z, yaw
			omegadot[0], # y, pitch
			omegadot[1]  # x, roll
		])

        self.omega = self.omega + dt * omegadot

        thetadot = eulerRateVelocity(self.theta, self.omega)
        self.theta = self.theta + dt * thetadot