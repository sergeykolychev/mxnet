# profiler setting methods
#

#' @export
MX.PROF.STATE <- list(STOP = 0L, RUN = 1L)

#' Set up the configuration of profiler.
#'
#' @param mode Indicting whether to enable the profiler, can be 'MX.PROF.MODE$SYMBOLIC' or 'MX.PROF.MODE$ALL'. Default is `MX.PROF.MODE$SYMBOLIC`.
#' @param filename The name of output trace file. Default is 'profile.json'
#'
#' @export
mx.profiler.config <- function(mode='symbolic', filename='profile.json', append_mode=0L) {
	mx.internal.profiler.config(mode, filename, append_mode)
}

#' Set up the profiler state to record operator.
#'
#' @param state  Indicting whether to run the profiler, can be 'MX.PROF.STATE$RUN' or 'MX.PROF.STATE$STOP'. Default is `MX.PROF.STATE$STOP`.
#' @param filename The name of output trace file. Default is 'profile.json'
#'
#' @export
mx.profiler.state <- function(state = MX.PROF.STATE$STOP) {
	mx.internal.profiler.state(state)
}
