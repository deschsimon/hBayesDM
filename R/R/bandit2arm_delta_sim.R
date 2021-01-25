#'Simulate data from bandit2arm_delta fit
#'
#' @description
#' Simulate data based on mu parameters of "hBayesDM" object fitted with
#' \code{bandit2arm_delta}
#'
#' @param b2a_d.fit "hBayesDM" object fitted with \code{bandit2arm_delta}
#'
#' @param seed seed for random number generation used to simulate the data
#'
#' @return A \code{data.frame} containing variables 'A', 'tau', 'PE', 'Q1',
#' 'Q2', 'p1', 'p2' in addition to original input data of fitted object.
#'
#' @export
#'
bandit2arm_delta_sim <- function(b2a_d.fit, seed = sample.int(.Machine$integer.max, 1)) {

  data <- b2a_d.fit$rawdata
  indPars <- b2a_d.fit$allIndPars
  subjIDs <- unique(data$subjID)
  numSubjs = length(subjIDs)
  initV = rep(0, 2)
  message('simulating data for ', b2a_d.fit$model, ' using data of ', numSubjs, 'subjects, seed: ', seed, '...')

  simAll <- c() # data.frame(subjID=NULL, trial=NULL, choice=NULL, PE=NULL, Q1=NULL, Q2=NULL, p1=NULL, p2=NULL)

  for (i in 1:numSubjs) {
    ID <- subjIDs[i]
    # message('## ', ID, ' ##')

    dat <- data%>%filter(subjID==ID)
    # message('data ID: ', toString(unique(dat$subjID)))
    Tsubj <- nrow(dat) #
    subjPars <- indPars%>%filter(subjID==ID)
    A <- subjPars%>%pull(A)
    tau <- subjPars%>%pull(tau)


    ev <- initV
    # log_lik <- c()

    simSub <- c()

    for (t in 1:Tsubj) {
      # message('trial ', t)
      probs <- softmax(ev*tau)
      choice <- sample(c(1, 2), size=1, replace = T, prob = probs)
      # outcome in trial t given choice
      outcome_choice <- ifelse(dat$outcome[t]==1&dat$choice[t]==1, 1, 2)
      PE <- outcome_choice - ev[choice]
      ev[choice] <- ev[choice] + A * PE

      simSub <- rbind(simSub, c(ID, t, choice, outcome_choice, A, tau, PE, ev[1], ev[2], probs[1], probs[2]))
    }

    simAll <- rbind(simAll, simSub)

  }

  simAll <- as.data.frame(simAll)
  colnames(simAll) <- c('subjID', 'trial', 'choice', 'outcome', 'A', 'tau', 'PE', 'Q1', 'Q2', 'p1', 'p2')
  return(simAll)
}
