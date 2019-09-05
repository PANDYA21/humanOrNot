local({
  r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos=r)
})

# custom require function that installs dependencies if not installed already and then requires them
require <- function(...) {
  success <- base::require(...)
  if (!success) {
    tryCatch(
      expr = {
        install.packages(...)
        base::require(...)
      },
      error = function(e) {
        e
      },
      finally = gc() # garbage collection
    )
  }
}

require('data.table')
setwd("/Users/bhaumik/github/bcg/humanOrNot")
heroes_info <- fread('data/heroes_information.csv')
hero_powers <- fread('data/super_hero_powers.csv')

joined <- merge(heroes_info, hero_powers, by.x = 'name', by.y = 'hero_names')