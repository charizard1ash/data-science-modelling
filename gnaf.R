### Load libraries ###
library(lubridate)
library(data.table)
library(parallel)


### Set variables ###
geo_location <- "" # location of file downloaded from https://data.gov.au/dataset/ds-dga-19432f89-dc3a-4ef3-b943-5326ef1dbecc/details?q=
no_cores <- detectCores()


### Import data ###
## GNAF.
# gnaf empty data.table.
gnaf <- data.table(GNAF_ID=NA)[0]

# File names.
gnaf_files <- unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=NULL, list=TRUE)
setDT(gnaf_files)

# geographical coordinates.
gnaf_geo <- mclapply(gnaf_files[grepl("/.*._address_default_geocode_psv.psv$", tolower(Name))==TRUE, Name],
                          function(x1) {
                            dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                            setDT(dt_1)
                            dt_1[, c(names(dt_1[, !c("ADDRESS_DETAIL_PID","LONGITUDE","LATITUDE")])):=NULL]
                            return(dt_1)
                          },
                     mc.cores=no_cores)
gnaf_geo <- rbindlist(l=gnaf_geo, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_geo, by.x=c("GNAF_ID"), by.y=c("ADDRESS_DETAIL_PID"), all=TRUE)
rm(gnaf_geo)

# detail.
gnaf_dtl <- mclapply(gnaf_files[grepl("/.*._address_detail_psv.psv$", tolower(Name))==TRUE, Name],
                     function(x1) {
                       dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                       setDT(dt_1)
                       dt_1[, c(names(dt_1[, !c("ADDRESS_DETAIL_PID","DATE_CREATED","BUILDING_NAME","LOT_NUMBER","LEVEL_NUMBER","FLAT_NUMBER","NUMBER_FIRST","NUMBER_LAST","STREET_LOCALITY_PID","LOCALITY_PID","POSTCODE","CONFIDENCE")])):=NULL]
                       dt_1[, c(names(dt_1)):=lapply(.SD, function(x) ifelse(is.character(x) & trimws(x)=="", NA, x)), .SDcols=names(dt_1)]
                       dt_1[, DATE_CREATED:=as.Date(DATE_CREATED, "%Y-%m-%d")]
                       dt_1[, POSTCODE:=paste0("0000", POSTCODE)]
                       dt_1[, POSTCODE:=substr(x=POSTCODE, start=nchar(POSTCODE)-3, stop=nchar(POSTCODE))]
                       return(dt_1)
                     },
                     mc.cores=no_cores)
gnaf_dtl <- rbindlist(l=gnaf_dtl, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_dtl, by.x=c("GNAF_ID"), by.y=c("ADDRESS_DETAIL_PID"), all=TRUE)
rm(gnaf_dtl)

# census meshblock 2016.
gnaf_mb16 <- mclapply(gnaf_files[grepl("*/.*._address_mesh_block_2016_psv.psv$", tolower(Name))==TRUE, Name],
                      function(x1) {
                        dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                        setDT(dt_1)
                        dt_1[, c(names(dt_1[, !c("ADDRESS_DETAIL_PID","MB_2016_PID")])):=NULL]
                        return(dt_1)
                      },
                      mc.cores=no_cores)
gnaf_mb16 <- rbindlist(l=gnaf_mb16, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_mb16, by.x=c("GNAF_ID"), by.y=c("ADDRESS_DETAIL_PID"), all=TRUE)
rm(gnaf_mb16)

# street locality name.
gnaf_strt_loc <- mclapply(gnaf_files[grepl("*/.*._street_locality_psv.psv$", tolower(Name))==TRUE, Name],
                          function(x1) {
                            dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                            setDT(dt_1)
                            dt_1[, c(names(dt_1[, !c("STREET_LOCALITY_PID","STREET_NAME","STREET_TYPE_CODE")])):=NULL]
                          },
                          mc.cores=no_cores)
gnaf_strt_loc <- rbindlist(l=gnaf_strt_loc, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_strt_loc, by.x=c("STREET_LOCALITY_PID"), by.y=c("STREET_LOCALITY_PID"), all.x=TRUE)
rm(gnaf_strt_loc)

# locality name.
gnaf_loc_nm <- mclapply(gnaf_files[grepl("*/.*._locality_psv.psv$", tolower(Name))==TRUE & grepl("_street_", tolower(Name))==FALSE, Name],
                        function(x1) {
                          dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                          setDT(dt_1)
                          dt_1[, c(names(dt_1[, !c("LOCALITY_PID","LOCALITY_NAME","STATE_PID")])):=NULL]
                        },
                        mc.cores=no_cores)
gnaf_loc_nm <- rbindlist(l=gnaf_loc_nm, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_loc_nm, by.x=c("LOCALITY_PID"), by.y=c("LOCALITY_PID"), all.x=TRUE)
rm(gnaf_loc_nm)

# state.
gnaf_state <- mclapply(gnaf_files[grepl("*/.*._state_psv.psv$", tolower(Name))==TRUE, Name],
                       function(x1) {
                         dt_1 <- read.delim(file=unzip(zipfile=paste0(geo_location, "nov19_gnaf_pipeseparatedvalue.zip"), files=x1), sep="|", header=TRUE, check.names=FALSE, stringsAsFactors=FALSE)
                         setDT(dt_1)
                         dt_1[, c(names(dt_1[, !c("STATE_PID","STATE_NAME","STATE_ABBREVIATION")])):=NULL]
                       },
                       mc.cores=no_cores)
gnaf_state <- rbindlist(l=gnaf_state, use.names=TRUE, fill=TRUE)
gnaf <- merge(gnaf, gnaf_state, by.x=c("STATE_PID"), by.y=c("STATE_PID"), all.x=TRUE)
rm(gnaf_state)

# remove unnecessary objects.
rm(gnaf_files)
gc(reset=TRUE)


### Export data ###
fwrite(x=gnaf, file=paste0(geo_location, "gnaf.txt"), sep="\t", quote=FALSE, row.names=FALSE, na="")
