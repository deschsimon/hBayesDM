#' Get combined table of posterior predictions from bandit2arm_delta
#'
#' @description
#' After running \code{pst_gainloss_Q_eo} with \code{inc_postpred = T}
#' the fitted "hBayesDM" object contains \code{y_pred}, \code{PE_pred}, and \code{ev_pred}.
#' This functions restructures this data to a single 2D \code{data.table}.
#'
#' @param fit A "hBayesDM" object fitted with \code{pst_gainloss_Q_eo} function.
#'
#' @return A \code{data.table} object.
#'
#' @export

pst_gainloss_Q_eo_pred <- function(fit) {
  r.d <- data.table(fit$rawdata)
  dims <- dim(fit$parVals$y_pred)
  n.iter <- dims[1]
  tsubj <- dims[3]
  r.d[, trial := 1:.N, by='subjID']
  f.d <- expand.grid(subjID=unique(r.d$subjID), trial=1:tsubj)
  i.d <- merge(r.d, f.d, all.y = T)
  fit.pred <- data.table::rbindlist(lapply(1:n.iter, function(i){
    iter.dat <- data.table(
      cbind(i.d,
            matrix(t(fit$parVals$y_pred[i, , ]), ncol=1, byrow = T),
            matrix(t(fit$parVals$PE_pred[i, , ]), ncol=1, byrow = T),
            matrix(t(fit$parVals$ev_pred[i, , , 1]), ncol=1, byrow = T),
            matrix(t(fit$parVals$ev_pred[i, , , 2]), ncol=1, byrow = T),
            rep(fit$parVals$alpha_pos[i, ], each=tsubj),
            rep(fit$parVals$alpha_neg[i, ], each=tsubj),
            rep(fit$parVals$beta[i, ], each=tsubj),
            i
      )
    )
    names(iter.dat) <- c(names(i.d), 'y_pred', 'PE_pred', 'ev1_pred', 'ev2_pred', 'alpha_pos', 'alpha_neg', 'beta', 'iter')
    iter.dat
  }))

  return(na.omit(fit.pred, cols='choice'))
}
