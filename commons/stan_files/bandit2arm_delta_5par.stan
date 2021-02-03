#include /pre/license.stan

data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=-1, upper=2> choice[N, T];
  real outcome[N, T];  // no lower and upper bounds
  
  // prior distribution parameters
  real mu_pr_m;
  real mu_pr_sd;
  real Arew_pr_m;
  real Arew_pr_sd;
  real Apun_pr_m;
  real Apun_pr_sd;
  real tau_pr_m;
  real tau_pr_sd;
  real R_pr_m;
  real R_pr_sd;
  real P_pr_m;
  real P_pr_sd;
  real sigma_m;
  real sigma_sd;
}
transformed data {
  vector[2] initV;  // initial values for EV
  initV = rep_vector(0.0, 2);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  vector[6] mu_pr;
  vector<lower=0>[6] sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] Arew_pr;    // reward learning rate
  vector[N] Apun_pr;    // punishment learning rate
  vector[N] tau_pr;     // inverse temperature
  vector[N] R_pr;       // reward sensitivity
  vector[N] P_pr;       // punishment sensitivity
}
transformed parameters {
  // subject-level parameters
  vector<lower=0, upper=1>[N] Arew;
  vector<lower=0, upper=1>[N] Apun;
  vector<lower=0, upper=5>[N] tau;
  vector<lower=0, upper=30>[N] R;
  vector<lower=0, upper=30>[N] P;

  Arew   = Phi_approx(mu_pr[1] + sigma[1] * Arew_pr);
  Apun   = Phi_approx(mu_pr[2] + sigma[2] * Apun_pr);
  tau    = Phi_approx(mu_pr[3] + sigma[3] * tau_pr) * 5;
  R      = Phi_approx(mu_pr[4] + sigma[4] * R_pr) * 30;
  P      = Phi_approx(mu_pr[5] + sigma[5] * P_pr) * 30;
}
model {
  // Hyperparameters
  mu_pr  ~ normal(mu_pr_m, mu_pr_sd);
  sigma ~ normal(sigma_m, sigma_sd);

  // individual parameters
  Arew_pr   ~ normal(Arew_pr_m, Arew_pr_sd);
  Apun_pr   ~ normal(Apun_pr_m, Apun_pr_sd);
  tau_pr    ~ normal(tau_pr_m,  tau_pr_sd);
  R_pr      ~ normal(R_pr_m,    R_pr_sd);
  P_pr      ~ normal(P_pr_m,    P_pr_sd);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[2] ev;    // expected value
    real PE;         // prediction error
    real alpha;      // effective learning rate (Arew or Apun, respectively)
    real sensitivity;// effective sensitivity (R or P, respectively)
    ev = initV;

    for (t in 1:(Tsubj[i])) {
      // compute action probabilities
      choice[i, t] ~ categorical_logit(tau[i] * ev);

      // prediction error
      sensitivity = (outcome[i, t] == 1 ? R[i] :  P[i]);
      PE = (outcome[i, t] * sensitivity) - ev[choice[i, t]];
      
      // value updating (learning)
      alpha = (PE >= 0) ? Arew[i] : Apun[i];
      ev[choice[i, t]] += alpha * PE;
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_Arew;
  real<lower=0, upper=1> mu_Apun;
  real<lower=0, upper=5> mu_tau;
  real<lower=0, upper=30> mu_R;
  real<lower=0, upper=30> mu_P;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];
  real PE_pred[N, T];
  real ev_pred[N, T, 2];

  // Set all posterior predictions to 0 (avoids NULL values)
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
      PE_pred[i, t] = -1;
      ev_pred[i, t, 1] = -1;
      ev_pred[i, t, 2] = -1;
    }
  }

  mu_Arew   = Phi_approx(mu_pr[1]);
  mu_Apun   = Phi_approx(mu_pr[2]);
  mu_tau    = Phi_approx(mu_pr[3]) * 5;
  mu_R      = Phi_approx(mu_pr[4]) * 30;
  mu_P      = Phi_approx(mu_pr[5]) * 30;

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[2] ev; // expected value
      real PE;      // prediction error
      int co;
      real alpha;
      real sensitivity;

      // Initialize values
      ev = initV;
      ev_pred[i, 1, 1] = 0;
      ev_pred[i, 1, 2] = 0;
      log_lik[i] = 0;

      for (t in 1:(Tsubj[i])) {
        // get choice of current trial (either 1 or 2)
        co = choice[i, t];
        
        // compute log likelihood of current trial
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | tau[i] * ev);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(softmax(tau[i] * ev));

        // prediction error
        sensitivity = (outcome[i, t] == 1 ?  R[i] : P[i]);
        PE = (outcome[i, t] * sensitivity) - ev[choice[i, t]];
        PE_pred[i, t] = PE;

        
        // value updating (learning)
        alpha = (PE >= 0) ? Arew[i] : Apun[i];
        ev[co] += alpha * PE;
        if (t > 1) {
           ev_pred[i, t] = ev_pred[i, t-1];
        }
        ev_pred[i, t, co] += alpha * PE;
      }
    }
  }
}

