#'Softmax function for matrices as defined for Stan
#'
#' @description
#' calculate softmax according to Stan definition
#'
#' @param probs vector of numbers
#'
#' @return A vector of same length as \code{probs} containing
#' the respective probabilities for each input element.
#'
#' @export
#'
softmax <- function(probs){
  if(is.null(dim(probs))) probs <- matrix(probs,ncol= length(probs))
  exp(probs)/apply(probs,1, function(x) sum(exp(x)))
}
