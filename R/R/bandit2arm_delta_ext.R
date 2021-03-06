#' @templateVar MODEL_FUNCTION bandit2arm_delta_ext
#' @templateVar CONTRIBUTOR 
#' @templateVar TASK_NAME 2-Armed Bandit Task
#' @templateVar TASK_CODE bandit2arm
#' @templateVar TASK_CITE (Erev et al., 2010; Hertwig et al., 2004)
#' @templateVar MODEL_NAME Rescorla-Wagner (Delta) Model (extended)
#' @templateVar MODEL_CODE delta_ext
#' @templateVar MODEL_CITE 
#' @templateVar MODEL_TYPE Hierarchical
#' @templateVar DATA_COLUMNS "subjID", "choice", "outcome"
#' @templateVar PARAMETERS \code{A} (learning rate), \code{tau} (inverse temperature)
#' @templateVar REGRESSORS 
#' @templateVar POSTPREDS "y_pred", "PE_pred", "ev_pred"
#' @templateVar LENGTH_DATA_COLUMNS 3
#' @templateVar DETAILS_DATA_1 \item{subjID}{A unique identifier for each subject in the data-set.}
#' @templateVar DETAILS_DATA_2 \item{choice}{Integer value representing the option chosen on the given trial: 1 or 2.}
#' @templateVar DETAILS_DATA_3 \item{outcome}{Integer value representing the outcome of the given trial (where reward == 1, and loss == -1).}
#' @templateVar LENGTH_ADDITIONAL_ARGS 1
#' @templateVar ADDITIONAL_ARGS_1 \item{priors}{\code{list()} object of floating point values representing the parameters of the prior distributions. For hyperparameters define \code{mu_pr_m} and \code{mu_pr_sd} for group mean and \code{sigma_m} and \code{sigma_sd} for the variance. For each parameter modelled define its mean by appending '\code{_m} and its sd by appending \code{_sd} to its name.}
#' 
#' @template model-documentation
#'
#' @export
#' @include hBayesDM_model.R
#' @include preprocess_funcs.R
#'
#' @references
#' Erev, I., Ert, E., Roth, A. E., Haruvy, E., Herzog, S. M., Hau, R., et al. (2010). A choice prediction competition: Choices from experience and from description. Journal of Behavioral Decision Making, 23(1), 15-47. http://doi.org/10.1002/bdm.683
#'
#' Hertwig, R., Barron, G., Weber, E. U., & Erev, I. (2004). Decisions From Experience and the Effect of Rare Events in Risky Choice. Psychological Science, 15(8), 534-539. http://doi.org/10.1111/j.0956-7976.2004.00715.x
#'

bandit2arm_delta_ext <- hBayesDM_model(
  task_name       = "bandit2arm",
  model_name      = "delta_ext",
  model_type      = "",
  data_columns    = c("subjID", "choice", "outcome"),
  parameters      = list(
    "A" = c(0, 0.5, 1),
    "tau" = c(0, 1, 5)
  ),
  regressors      = NULL,
  postpreds       = c("y_pred", "PE_pred", "ev_pred"),
  preprocess_func = bandit2arm_preprocess_func)
