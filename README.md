# StochasticOptimalControl_ExtendedvsUnscented_Approach
This repository is dedicated to a comparison between two approaches (Extended Transform; Unscented Transform) to perform stochastic optimal control simulations. Both approaches perform an approximation converting the stochastic into an augmented deterministic problem. 

Extended transform:
- The generally non-normally distributed state trajectories are approximated as normally distributed. This allows describing the full state space by the mean state trajectory and the state co-variance trajectory.
- The propagation of the mean state is according to the deterministic version of the stochastic dynamics (all noise sources are set to zero).
- The propagation of the state covariance is according to the rules of an Extended Kalman Filter (linearization of the dynamics around the operating point throughout the state trajectory).

Unscented transform:
- The generally non-normally distributed state trajectories are approximated as normally distributed. This allows describing the full state space by the mean state trajectory and the state co-variance trajectory.
- The propagation of the mean state trajectory and state covariance is performed based on the Unscented Transform. Each time-integration step exists of (1) deterministic sampling around the current mean state according to the current covariance, (2) computing the time-integration over one time-interval of these samples = posterior samples (non-linear operation), (3) computing the posterior mean and covariance from the posterior samples.

Both approaches result in a deterministic optimal control problem that we discretize into an NLP using direct collocation.
The unscented transform approach results in more accurate predictive simulations, that are closer to the true solution of the stochastic optimal control problem. 

To make an easier comparison where the focus is on the approximation of the non-linear effects, we chose to optimize a non-linear system for one specific operating point; no time-integration.
We simulate an inverted pendulum model (q ̈=-g.l.sin(q)+T_total) actuated by two antagonistic ideal torque actuators determined by constant feedforward torque and constant PD feedback: T^+=T_base^++K_p^+ (q-π)+K_v^+ q ̇; T^-=T_base^-+K_p^- (q-π)+K_v^- q ̇. The total torque (T_total) applied to the IP is corrupted by motor noise and clipped between 0 and 250Nm with a smooth but accurate approximation of a clipping function: T_total=f_clip (T^+ )-f_clip (T^- )+T_noise), with T_noise zero-mean Gaussian noise with standard deviation σ_noise. We solve for  p=[T_base^+,T_base^-,K^+,B^+,K^-,B^-] that minimize the expected effort (integral of T_total squared) and achieve an upright equilibrium posture: [q,q ̇,q ̈]=0, that is stable: P_k=P_(k+1). 
