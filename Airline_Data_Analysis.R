library(tidyverse)
library(data.table)
library(lubridate)
library(skimr)
library(timetk)
library(highcharter)
library(h2o)
library(tidymodels)
library(modeltime)
library(inspectdf)



#You have been given Box & Jenking airline data set with total number of international airline
#passengers from 1949 to 1960 and have been asked to use three models to fit the data, test their
#performance and make forecast:




df <- fread('AirPassengers (3).csv')
df %>% view()
colnames(df) <- c('Date','Count')
lapply(df,class)
df %>% inspect_na()
df <- df %>% mutate(Date=paste(Date,'-01',sep=''))
df$Date <- df$Date %>% as.Date('%Y-%m-%d')




df %>% plot_time_series(Date,Count,                               
                        .color_var = lubridate::year(Date),
                        .color_lab = "Year",
                        .interactive=T, .plotly_slider = T)


df %>% plot_seasonal_diagnostics(Date,Count,.interactive=T)

df %>%  plot_acf_diagnostics(Date,Count,.lags='1 year', .interactive = T)

all_time_arg <- df %>% tk_augment_timeseries_signature(Date)

all_time_arg %>% skim


df <- all_time_arg %>% select(-contains('hour'),
                                -contains('week'),
                                -contains('day'),
                                -minute,-am.pm,-second) %>% 
  mutate_if(is.ordered,as.character) %>% mutate_if(is.character,as.factor)


#Modelling----------------------------------------------------------------------

h2o.init()

train_air <- df %>% filter(year < 1957) %>% as.h2o()
test_air <- df %>% filter(year >= 1957) %>% as.h2o()

y <- 'Count'

x <- df %>% select(-Count) %>% names()

model_h2o <- h2o.automl(
  x = x, y = y, 
  training_frame = train_air, 
  validation_frame = test_air,
  leaderboard_frame = test_air,
  stopping_metric = "RMSE",
  seed = 123, nfolds = 10,
  exclude_algos = "GLM",
  max_runtime_secs = 120) 

model_h2o@leaderboard %>% as.data.frame()
h2o_leader <- model_h2o@leader

pred_h2o <- h2o_leader %>% h2o.predict(test_air)

h2o_leader %>% h2o.rmse(train=T,valid=T,xval = T)

error_tbl <- df %>% 
  filter(lubridate::year(Date) >= 1957) %>% 
  add_column(pred = pred_h2o %>% as_tibble() %>% pull(predict)) %>%
  rename(actual = Count) %>% 
  select(Date,actual,pred)

highchart() %>% 
  hc_xAxis(categories = error_tbl$Date) %>% 
  hc_add_series(data=error_tbl$actual, type='line', color='red', name='Actual') %>% 
  hc_add_series(data=error_tbl$pred, type='line', color='green', name='Predicted') %>% 
  hc_title(text='Predict')



  
  new_data <- seq(as.Date('1961-01-01'),as.Date('1962-12-01'),'months') %>% 
  as.tibble() %>% add_column(Count=0) %>% 
  rename(Date=value) %>% 
  tk_augment_timeseries_signature() %>% 
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)

new_data %>% view()

new_h2o <- new_data %>% as.h2o()

new_predictions <- h2o_leader %>% 
  h2o.predict(new_h2o) %>% 
  as_tibble() %>%
  add_column(Date=new_data$Date) %>% 
  select(Date,predict) %>% 
  rename(Count=predict)

new_predictions %>% dim()
df %>% dim()

df %>% 
  bind_rows(new_predictions) %>% 
  mutate(colors=c(rep('Actual', 144),rep('Predicted', 24))) %>% 
  hchart("line", hcaes(Date, Count, group = colors)) %>% 
  hc_title(text='Forecast') %>% 
  hc_colors(colors = c('red','green'))





#1. Using arima_boost(), exp_smoothing(), prophet_reg() models;

train_f <- df %>% filter(Date<'1957-01-01') %>%  as.data.frame()
test_f <- df %>% filter(Date>='1957-01-01') %>%  as.data.frame()

#Arima
model_fit_arima_b <- arima_boost() %>% set_engine('auto_arima_xgboost') %>% 
  fit(Count~Date,train_f)



#prophet
model_fit_prophet <- prophet_reg(seasonality_yearly = T) %>% set_engine('prophet') %>% 
  fit(Count~Date,train_f)

# exp_smoothing

model_fit_exp <- exp_smoothing() %>% set_engine('ets') %>% fit(Count~Date,train_f)


calibration <- modeltime_table(
  model_fit_arima_b,
  model_fit_prophet,
  model_fit_exp) %>% modeltime_calibrate(test_f)


calibration %>% 
  modeltime_forecast(actual_data = df) %>%
  plot_modeltime_forecast(.interactive = T,
                          .plotly_slider = T)  

# Comparing RMSE scores on test set

calibration %>% modeltime_accuracy() %>% 
  table_modeltime_accuracy(.interactive = F)

#According to accuracy results , the prediction of Arima model shows  the lowest RMSE. This means
#we forecast according to Arima and it is in the first place


#Visualizing past data and forecast values on one plot; make separation with two different colors.

calibration %>%
  filter(.model_id %in% 1) %>% 
  modeltime_refit(df) %>%
  modeltime_forecast(h = "2 year", 
                     actual_data = df) %>%
  select(-contains("conf")) %>% 
  plot_modeltime_forecast(.interactive = T,
                          .plotly_slider = T,
                          .legend_show = F)


