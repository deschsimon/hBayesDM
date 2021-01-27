#' Get combined table of posterior predictions from bandit2arm_delta
#'
#' @description
#' After running \code{bandit2arm_delta} with \code{inc_postpred = T}
#' the fitted "hBayesDM" object contains \code{y_pred}, \code{PE_pred}, and \code{ev_pred}.
#' This functions restructures this data to a single 2D \code{data.table}.
#'
#' @param fit A "hBayesDM" object fitted with \code{bandit2arm_delta} function.
#'
#' @return A \code{data.table} object.
#'
#' @export

bandit2arm_delta_pred <- function(fit) {
    raw.dat <- fit$rawdata
    st <- data.table(raw.dat)%>%.[, (n=.N), by='subjID']
    subj.trials <- st[[2]]
    n.iter <- dim(fit$parVals$y_pred)[1]
    fit.pred <- data.table::rbindlist(lapply(1:n.iter, function(i){
        iter.dat <- raw.dat
        y_preds <- na.omit(as.numeric(t(fit$parVals$y_pred[i, , ])))
        y_preds_inlvalid <- attr(y_preds, 'na.action')
        iter.dat$y_pred <- y_preds
        iter.dat$PE_pred <- as.numeric(t(fit$parVals$PE_pred[i, , ]))[-y_preds_inlvalid]
        iter.dat$ev1_pred <- na.omit(as.numeric(t(fit$parVals$ev_pred[i, , , 1])[-y_preds_inlvalid]))
        iter.dat$ev2_pred <- na.omit(as.numeric(t(fit$parVals$ev_pred[i, , , 2])[-y_preds_inlvalid]))
        iter.dat$A <- rep(fit$parVals$A[i, ], subj.trials)
        iter.dat$tau <- rep(fit$parVals$tau[i, ], subj.trials)
        iter.dat$iter <- i
        iter.dat
    }))
    return(fit.pred)
}
