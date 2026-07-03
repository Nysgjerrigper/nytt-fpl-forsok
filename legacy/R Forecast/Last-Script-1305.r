# Forecasts for FPL Script to Master thesis EBA&PAM

rm(list = ls(all = TRUE))

## 0: Preamble ----
Sys.setenv(WORKON_HOME = "C:/Users/peram/Documents/test/R Forecast/.virtualenvs")
reticulate::use_virtualenv("C:/Users/peram/Documents/test/R Forecast/.virtualenvs/r-reticulate", required = TRUE)

library(reticulate)
library(keras)

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggthemes, tidyverse, slider,
               glmnet,httr,jsonlite,tensorflow,
               Metrics, pastecs, stats, png, grid)

set.seed(1)  
tensorflow::set_random_seed(1)

# Lists to store results ----
model_list       <- list()
scaling_factors  <- list()
predictions_list <- list()
metrics_list     <- list() 
validation_metrics <- list() 

# Global user inputs ----
split_gw <- 38+38
epoker <- 15
vindu <- 3
emb_dim <- 16
num_patience <- round(epoker*0.5)

# Fetch data and prelim data manipulation ----
df <- read_csv("C:/Users/peram/Documents/test/Datasett/Ekstra kolonner, stigende GW, alle tre sesonger(22-24), heltall.csv")

## Positions Partitions ----
gk <- df %>%
  filter(position == "GK") %>%
  mutate(
    row_id  = row_number(),
    pID_idx = as.integer(factor(player_id)),
    tID_idx = as.integer(factor(tID)),
    oID_idx = as.integer(factor(oID))
  )
def <- df %>%
  filter(position == "DEF") %>%
  mutate(
    row_id  = row_number(),
    pID_idx = as.integer(factor(player_id)),
    tID_idx = as.integer(factor(tID)),
    oID_idx = as.integer(factor(oID))
  )
mid <- df %>%
  filter(position == "MID") %>%
  mutate(
    row_id  = row_number(),
    pID_idx = as.integer(factor(player_id)),
    tID_idx = as.integer(factor(tID)),
    oID_idx = as.integer(factor(oID))
  )
fwd <- df %>%
  filter(position == "FWD") %>%
  mutate(
    row_id  = row_number(),
    pID_idx = as.integer(factor(player_id)),
    tID_idx = as.integer(factor(tID)),
    oID_idx = as.integer(factor(oID))
  )

unscaled_gk <- gk
unscaled_def <- def
unscaled_mid <- mid
unscaled_fwd <- fwd


# 1: Goalkeepers (GK) ----

## GK 1.1 Define Features, Target & Scale ----
numF <- c(
  "assists","creativity","minutes","goals_conceded","saves","bonus","bps",
  "expected_assists","expected_goal_involvements","expected_goals","ict_index",
  "own_goals","red_cards","threat","transfers_in","transfers_out","yellow_cards",
  "expected_goals_conceded","penalties_saved","value","selected",
  "transfers_balance","starts","influence","clean_sheets"
)
tar <- "total_points"
initial_numF_gk <- length(numF)

gk <- gk |> arrange(pID_idx, GW)

mu <- mean(gk[[tar]], na.rm = TRUE)
sigma <- sd(gk[[tar]], na.rm = TRUE)

gk <- gk %>%
  mutate(across(all_of(numF), ~ (.x - mean(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE)),
         !!tar := (.data[[tar]] - mu) / sigma)

## GK 1.2: Build rolling numerical windows ----
ws <- vindu
numW <- gk %>%
  group_by(pID_idx) %>%
  filter(n() > ws) %>%
  group_modify(~{
    W <- slide(.x[, numF], .f = ~as.matrix(.x), .before = ws, .after = -1, .complete = TRUE)
    valid <- W[(ws + 1):length(W)]
    tgt <- .x[[tar]][(ws + 1):nrow(.x)]
    rid <- .x$row_id[(ws + 1):nrow(.x)]
    tibble(window = valid, target = tgt, row_id = rid)
  }) %>%
  ungroup() %>%
  left_join(select(gk, row_id, GW), by = "row_id")

## GK 1.3: Lasso Regression ----
Xl <- do.call(rbind, lapply(numW$window, colMeans, na.rm = TRUE))
yl <- numW$target
cvfit <- cv.glmnet(Xl, yl, alpha = 1)
minlambda <- cvfit$lambda.min
lassofit <- glmnet(Xl, yl, alpha = 1, lambda = minlambda)
keepF <- rownames(as.matrix(coef(lassofit)))[coef(lassofit)[, 1] != 0] %>% setdiff("(Intercept)")
numF_gk <- keepF
numF <- keepF # Update numF globally for this position block

## GK 1.4.1: Rebuild Numeric Windows ----
ws <- vindu
numW <- gk %>%
  group_by(pID_idx) %>%
  filter(n() > ws) %>%
  group_modify(~{
    W <- slide(.x[, numF], .f = ~as.matrix(.x), .before = ws, .after = -1, .complete = TRUE)
    valid <- W[(ws + 1):length(W)]
    tgt <- .x[[tar]][(ws + 1):nrow(.x)]
    rid <- .x$row_id[(ws + 1):nrow(.x)]
    tibble(window = valid, target = tgt, row_id = rid)
  }) %>%
  ungroup() %>%
  left_join(select(gk, row_id, GW), by = "row_id")

### GK 1.4.2: Build & align categorical windows ----
catW_all <- gk %>%
  group_by(player_id) %>%
  filter(n() > ws) %>%
  group_modify(~{
    W <- slide(.x %>% select(pID_idx, tID_idx, oID_idx, hID), .f = ~as.matrix(.x), .before = ws, .after = -1, .complete = TRUE)
    valid <- W[(ws + 1):length(W)]
    rid <- .x$row_id[(ws + 1):nrow(.x)]
    tibble(window = valid, row_id = rid)
  }) %>%
  ungroup()
catW <- numW %>%
  select(row_id) %>%
  left_join(catW_all, by = "row_id")

### GK 1.4.3: To Arrays ----
nSamp <- nrow(numW)
num_array <- array(unlist(numW$window), dim = c(nSamp, ws, length(numF)))
targets <- matrix(numW$target, ncol = 1)
pID_hist_array <- t(sapply(catW$window, function(m) m[, "pID_idx"]))
tID_hist_array <- t(sapply(catW$window, function(m) m[, "tID_idx"]))
oID_hist_array <- t(sapply(catW$window, function(m) m[, "oID_idx"]))
hID_hist_array <- t(sapply(catW$window, function(m) m[, "hID"]))
cat_cur_mat <- do.call(rbind, lapply(catW$window, function(m) m[nrow(m), ]))
pID_cur_array <- matrix(cat_cur_mat[, "pID_idx"], ncol = 1)
tID_cur_array <- matrix(cat_cur_mat[, "tID_idx"], ncol = 1)
oID_cur_array <- matrix(cat_cur_mat[, "oID_idx"], ncol = 1)
hID_cur_array <- matrix(cat_cur_mat[, "hID"], ncol = 1)

## GK 1.5: Split Data ----
idx <- which(numW$GW <= split_gw)
Xnum_train <- num_array[idx,,,drop=F]
Xnum_val <- num_array[-idx,,,drop=F]
y_train <- targets[idx,,drop=F]
y_val <- targets[-idx,,drop=F]
Xcat_hist_tr <- list(pID_hist=pID_hist_array[idx,,drop=F], tID_hist=tID_hist_array[idx,,drop=F], oID_hist=oID_hist_array[idx,,drop=F], hID_hist=hID_hist_array[idx,,drop=F])
Xcat_hist_va <- list(pID_hist=pID_hist_array[-idx,,drop=F], tID_hist=tID_hist_array[-idx,,drop=F], oID_hist=oID_hist_array[-idx,,drop=F], hID_hist=hID_hist_array[-idx,,drop=F])
Xcat_cur_tr <- list(pID_cur=pID_cur_array[idx,,drop=F], tID_cur=tID_cur_array[idx,,drop=F], oID_cur=oID_cur_array[idx,,drop=F], hID_cur=hID_cur_array[idx,,drop=F])
Xcat_cur_va <- list(pID_cur=pID_cur_array[-idx,,drop=F], tID_cur=tID_cur_array[-idx,,drop=F], oID_cur=oID_cur_array[-idx,,drop=F], hID_cur=hID_cur_array[-idx,,drop=F])

## GK 1.6: Building the Keras Neural Network Model ----
nP <- max(gk$pID_idx,na.rm=T); nT <- max(gk$tID_idx,na.rm=T); nO <- max(gk$oID_idx,na.rm=T)
inp_num <- layer_input(shape=c(ws,length(numF)),name="input_seq")
inp_ph  <- layer_input(shape=c(ws),dtype="int32",name="pID_hist")
inp_th  <- layer_input(shape=c(ws),dtype="int32",name="tID_hist")
inp_oh  <- layer_input(shape=c(ws),dtype="int32",name="oID_hist")
inp_hh  <- layer_input(shape=c(ws),dtype="int32",name="hID_hist")
inp_pc  <- layer_input(shape=c(1),dtype="int32",name="pID_cur")
inp_tc  <- layer_input(shape=c(1),dtype="int32",name="tID_cur")
inp_oc  <- layer_input(shape=c(1),dtype="int32",name="oID_cur")
inp_hc  <- layer_input(shape=c(1),dtype="int32",name="hID_cur")

emb_ph_hist_stream <- inp_ph %>% layer_embedding(input_dim=nP+1,output_dim=emb_dim,mask_zero=T)
emb_th_hist_stream <- inp_th %>% layer_embedding(input_dim=nT+1,output_dim=emb_dim,mask_zero=T)
emb_oh_hist_stream <- inp_oh %>% layer_embedding(input_dim=nO+1,output_dim=emb_dim,mask_zero=T)
emb_hh_hist_stream <- inp_hh %>% layer_lambda(function(x){tf$expand_dims(tf$cast(x,tf$float32),axis=as.integer(-1))})
cat_hist_seq_stream <- layer_concatenate(list(emb_ph_hist_stream,emb_th_hist_stream,emb_oh_hist_stream,emb_hh_hist_stream),axis=-1)

emb_pc_cur_stream <- inp_pc %>% layer_embedding(input_dim=nP+1,output_dim=emb_dim) %>% layer_flatten()
emb_tc_cur_stream <- inp_tc %>% layer_embedding(input_dim=nT+1,output_dim=emb_dim) %>% layer_flatten()
emb_oc_cur_stream <- inp_oc %>% layer_embedding(input_dim=nO+1,output_dim=emb_dim) %>% layer_flatten()
emb_hc_cur_stream <- inp_hc %>% layer_lambda(function(x)tf$cast(x,tf$float32))
seq_pc_cur_stream <- emb_pc_cur_stream %>% layer_repeat_vector(n=ws)
seq_tc_cur_stream <- emb_tc_cur_stream %>% layer_repeat_vector(n=ws)
seq_oc_cur_stream <- emb_oc_cur_stream %>% layer_repeat_vector(n=ws)
seq_hc_cur_stream <- emb_hc_cur_stream %>% layer_repeat_vector(n=ws)
current_cat_as_seq_stream <- layer_concatenate(list(seq_pc_cur_stream,seq_tc_cur_stream,seq_oc_cur_stream,seq_hc_cur_stream),axis=-1)

lstm_in_stream <- layer_concatenate(list(inp_num,cat_hist_seq_stream,current_cat_as_seq_stream),axis=-1)
lstm_out_stream <- lstm_in_stream %>% layer_lstm(units=64,return_sequences=F) %>% layer_dropout(rate=0.2)

head_stream <- lstm_out_stream %>% layer_dense(units=32,activation="relu")
out_stream <- head_stream %>% layer_dense(units=1,activation="linear")

model <- keras_model(inputs=list(inp_num,inp_ph,inp_th,inp_oh,inp_hh,inp_pc,inp_tc,inp_oc,inp_hc),outputs=out_stream,name="gk_lstm_model_v2")
model %>% compile(optimizer=optimizer_adam(),loss="mse",metrics=metric_root_mean_squared_error())

summary(model)
keras_python <- import("keras")
keras_python$utils$plot_model(model,to_file="keras-model-gk-v2.png",show_shapes=T,show_dtype=F,show_layer_names=T,rankdir="TB",expand_nested=F,dpi=200,show_layer_activations=F)
img <- readPNG("keras-model-gk-v2.png")
grid::grid.raster(img)

## GK 1.7: Callback Functions and Scaling Factors ----
weights_filepath_gk <- "R Forecast/best_gk_v2.hdf5"
early_stopping <- callback_early_stopping(monitor="val_loss",patience=num_patience,restore_best_weights=T)
model_checkpoint_gk <- callback_model_checkpoint(filepath=weights_filepath_gk,monitor="val_loss",save_best_only=T,save_weights_only=T,mode="min",verbose=1)
reduce_lr <- callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.2,patience=floor(num_patience*0.5),min_lr=1e-6,verbose=1)
scaling_factors$gk <- list(mu=mu,sigma=sigma,numF=numF)

## GK 1.8: Train Goalkeeper LSTM Model ----
history <- model %>% fit(
  x = c(list(input_seq=Xnum_train), Xcat_hist_tr, Xcat_cur_tr),
  y = y_train,
  epochs = epoker,
  batch_size = 16,
  validation_data = list(c(list(input_seq=Xnum_val), Xcat_hist_va, Xcat_cur_va), y_val),
  callbacks = list(early_stopping, model_checkpoint_gk, reduce_lr)
)

## GK 1.9: Generate Predictions & Invert Scaling ----
val_row_ids <- numW$row_id[-idx]
pred_all <- model %>% predict(c(list(input_seq=Xnum_val), Xcat_hist_va, Xcat_cur_va))
preds_gk <- tibble(
  row_id=val_row_ids,
  predicted_total_points_scaled=as.vector(pred_all),
  actual_total_points_scaled=as.vector(y_val)
) %>%
  mutate(
    predicted_total_points = predicted_total_points_scaled * sigma + mu,
    actual_total_points = actual_total_points_scaled * sigma + mu
  ) %>%
  left_join(unscaled_gk, by="row_id")

## GK 1.10 Save model and validation set predictions ----
model_list$gk <- model
preds_gk_clean <- preds_gk %>%
  select(row_id, GW, player_id, name, position, team, actual_total_points, predicted_total_points, value)

## GK 1.11: Plots for Evaluation ----
history_gk <- history
plot_history_gk <- plot(history_gk)
print(plot_history_gk)

plot_actual_predicted_gk <- ggplot(preds_gk, aes(x=actual_total_points, y=predicted_total_points)) +
  geom_point(alpha=0.5) +
  geom_abline(intercept=0, slope=1, linetype="dashed", color="red") +
  labs(title="GK Actual vs. Predicted", x="Actual", y="Predicted") +
  facet_wrap(~position) +
  theme_grey() +
  coord_cartesian(xlim=range(preds_gk$actual_total_points,na.rm=T), ylim=range(preds_gk$predicted_total_points,na.rm=T))
print(plot_actual_predicted_gk)

preds_gk_residuals <- preds_gk %>% mutate(residual=actual_total_points-predicted_total_points)
plot_residuals_predicted_gk <- ggplot(preds_gk_residuals, aes(x=predicted_total_points, y=residual)) +
  geom_point(alpha=0.5) +
  geom_hline(yintercept=0, linetype="dashed", color="red") +
  labs(title="GK Residuals vs. Predicted", x="Predicted", y="Residual") +
  facet_wrap(~position) +
  theme_grey()
print(plot_residuals_predicted_gk)

plot_distribution_residuals_gk <- ggplot(preds_gk_residuals, aes(x=residual)) +
  geom_histogram(aes(y=after_stat(density)), bins=30, fill="blue", alpha=0.7) +
  geom_density(color="red") +
  geom_vline(xintercept=0, linetype="dashed", color="black") +
  labs(title="GK Residual Distribution", x="Residual", y="Density") +
  facet_wrap(~position) +
  theme_grey()
print(plot_distribution_residuals_gk)

model_evaluation_gk <- model |> evaluate(c(list(input_seq=Xnum_val), Xcat_hist_va, Xcat_cur_va), y_val, verbose=0)
validation_metrics$gk <- model_evaluation_gk


# 2: Defenders (DEF) ----

## DEF 2.1 Define Features, Target & Scale ----
numF <- c("goals_scored","assists", "creativity", "minutes", "goals_conceded",
          "bonus", "bps", "expected_assists", "expected_goal_involvements",
          "expected_goals", "ict_index", "own_goals", "red_cards",
          "threat", "transfers_in", "transfers_out", "yellow_cards",
          "expected_goals_conceded", "value",
          "selected", "transfers_balance", "starts", "influence",
          "clean_sheets")
tar <- "total_points"
initial_numF_def <- length(numF)

def <- def |> arrange(pID_idx, GW)

mu <- mean(def[[tar]], na.rm=T)
sigma <- sd(def[[tar]], na.rm=T)

def <- def %>%
  mutate(across(all_of(numF),~(.x-mean(.x,na.rm=T))/sd(.x,na.rm=T)),
         !!tar:=(.data[[tar]]-mu)/sigma)

## DEF 2.2: Build rolling numerical windows ----
ws <- vindu
numW <- def %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(def,row_id,GW),by="row_id")

## DEF 2.3: Lasso Regression ----
Xl <- do.call(rbind,lapply(numW$window,colMeans,na.rm=T))
yl <- numW$target
cvfit <- cv.glmnet(Xl,yl,alpha=1)
minlambda <- cvfit$lambda.min
lassofit <- glmnet(Xl,yl,alpha=1,lambda=minlambda)
keepF <- rownames(as.matrix(coef(lassofit)))[coef(lassofit)[,1]!=0] %>% setdiff("(Intercept)")
numF_def <- keepF
numF <- keepF

## DEF 2.4.1: Rebuild Numeric Windows ----
ws <- vindu
numW <- def %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(def,row_id,GW),by="row_id")

### DEF 2.4.2: Build & align categorical windows ----
catW_all <- def %>%
  group_by(player_id) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x%>%select(pID_idx,tID_idx,oID_idx,hID),.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,row_id=rid)
  }) %>%
  ungroup()
catW <- numW %>%
  select(row_id) %>%
  left_join(catW_all,by="row_id")

### DEF 2.4.3: To Arrays ----
nSamp<-nrow(numW)
num_array<-array(unlist(numW$window),dim=c(nSamp,ws,length(numF)))
targets<-matrix(numW$target,ncol=1)
pID_hist_array<-t(sapply(catW$window,function(m)m[,"pID_idx"]))
tID_hist_array<-t(sapply(catW$window,function(m)m[,"tID_idx"]))
oID_hist_array<-t(sapply(catW$window,function(m)m[,"oID_idx"]))
hID_hist_array<-t(sapply(catW$window,function(m)m[,"hID"]))
cat_cur_mat<-do.call(rbind,lapply(catW$window,function(m)m[nrow(m),]))
pID_cur_array<-matrix(cat_cur_mat[,"pID_idx"],ncol=1)
tID_cur_array<-matrix(cat_cur_mat[,"tID_idx"],ncol=1)
oID_cur_array<-matrix(cat_cur_mat[,"oID_idx"],ncol=1)
hID_cur_array<-matrix(cat_cur_mat[,"hID"],ncol=1)

## DEF 2.5: Split Data ----
idx<-which(numW$GW<=split_gw)
Xnum_train<-num_array[idx,,,drop=F]
Xnum_val<-num_array[-idx,,,drop=F]
y_train<-targets[idx,,drop=F]
y_val<-targets[-idx,,drop=F]
Xcat_hist_tr<-list(pID_hist=pID_hist_array[idx,,drop=F],tID_hist=tID_hist_array[idx,,drop=F],oID_hist=oID_hist_array[idx,,drop=F],hID_hist=hID_hist_array[idx,,drop=F])
Xcat_hist_va<-list(pID_hist=pID_hist_array[-idx,,drop=F],tID_hist=tID_hist_array[-idx,,drop=F],oID_hist=oID_hist_array[-idx,,drop=F],hID_hist=hID_hist_array[-idx,,drop=F])
Xcat_cur_tr<-list(pID_cur=pID_cur_array[idx,,drop=F],tID_cur=tID_cur_array[idx,,drop=F],oID_cur=oID_cur_array[idx,,drop=F],hID_cur=hID_cur_array[idx,,drop=F])
Xcat_cur_va<-list(pID_cur=pID_cur_array[-idx,,drop=F],tID_cur=tID_cur_array[-idx,,drop=F],oID_cur=oID_cur_array[-idx,,drop=F],hID_cur=hID_cur_array[-idx,,drop=F])

## DEF 2.6: Building the Keras Neural Network Model ----
nP<-max(def$pID_idx,na.rm=T);nT<-max(def$tID_idx,na.rm=T);nO<-max(def$oID_idx,na.rm=T)
inp_num<-layer_input(shape=c(ws,length(numF)),name="input_seq")
inp_ph<-layer_input(shape=c(ws),dtype="int32",name="pID_hist")
inp_th<-layer_input(shape=c(ws),dtype="int32",name="tID_hist")
inp_oh<-layer_input(shape=c(ws),dtype="int32",name="oID_hist")
inp_hh<-layer_input(shape=c(ws),dtype="int32",name="hID_hist")
inp_pc<-layer_input(shape=c(1),dtype="int32",name="pID_cur")
inp_tc<-layer_input(shape=c(1),dtype="int32",name="tID_cur")
inp_oc<-layer_input(shape=c(1),dtype="int32",name="oID_cur")
inp_hc<-layer_input(shape=c(1),dtype="int32",name="hID_cur")

emb_ph_hist_stream<-inp_ph%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim,mask_zero=T)
emb_th_hist_stream<-inp_th%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim,mask_zero=T)
emb_oh_hist_stream<-inp_oh%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim,mask_zero=T)
emb_hh_hist_stream<-inp_hh%>%layer_lambda(function(x){tf$expand_dims(tf$cast(x,tf$float32),axis=as.integer(-1))})
cat_hist_seq_stream<-layer_concatenate(list(emb_ph_hist_stream,emb_th_hist_stream,emb_oh_hist_stream,emb_hh_hist_stream),axis=-1)

emb_pc_cur_stream<-inp_pc%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim)%>%layer_flatten()
emb_tc_cur_stream<-inp_tc%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim)%>%layer_flatten()
emb_oc_cur_stream<-inp_oc%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim)%>%layer_flatten()
emb_hc_cur_stream<-inp_hc%>%layer_lambda(function(x)tf$cast(x,tf$float32))
seq_pc_cur_stream<-emb_pc_cur_stream%>%layer_repeat_vector(n=ws)
seq_tc_cur_stream<-emb_tc_cur_stream%>%layer_repeat_vector(n=ws)
seq_oc_cur_stream<-emb_oc_cur_stream%>%layer_repeat_vector(n=ws)
seq_hc_cur_stream<-emb_hc_cur_stream%>%layer_repeat_vector(n=ws)
current_cat_as_seq_stream<-layer_concatenate(list(seq_pc_cur_stream,seq_tc_cur_stream,seq_oc_cur_stream,seq_hc_cur_stream),axis=-1)

lstm_in_stream<-layer_concatenate(list(inp_num,cat_hist_seq_stream,current_cat_as_seq_stream),axis=-1)
lstm_out_stream<-lstm_in_stream%>%layer_lstm(units=64,return_sequences=F)%>%layer_dropout(rate=0.2)

head_stream<-lstm_out_stream%>%layer_dense(units=32,activation="relu")
out_stream<-head_stream%>%layer_dense(units=1,activation="linear")

model<-keras_model(inputs=list(inp_num,inp_ph,inp_th,inp_oh,inp_hh,inp_pc,inp_tc,inp_oc,inp_hc),outputs=out_stream,name="def_lstm_model_v2")
model%>%compile(optimizer=optimizer_adam(),loss="mse",metrics=metric_root_mean_squared_error())

summary(model)
keras_python<-import("keras")
keras_python$utils$plot_model(model,to_file="keras-model-def-v2.png",show_shapes=T,show_dtype=F,show_layer_names=T,rankdir="TB",expand_nested=F,dpi=200,show_layer_activations=F)
img<-readPNG("keras-model-def-v2.png")
grid::grid.raster(img)

## DEF 2.7: Callback Functions and Scaling Factors ----
weights_filepath_def<-"R Forecast/best_def_v2.hdf5"
early_stopping<-callback_early_stopping(monitor="val_loss",patience=num_patience,restore_best_weights=T)
model_checkpoint_def<-callback_model_checkpoint(filepath=weights_filepath_def,monitor="val_loss",save_best_only=T,save_weights_only=T,mode="min",verbose=1)
reduce_lr<-callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.2,patience=floor(num_patience*0.5),min_lr=1e-6,verbose=1)
scaling_factors$def<-list(mu=mu,sigma=sigma,numF=numF)

## DEF 2.8: Train Defender LSTM Model ----
history<-model%>%fit(
  x=c(list(input_seq=Xnum_train),Xcat_hist_tr,Xcat_cur_tr),
  y=y_train,
  epochs=epoker,
  batch_size=16,
  validation_data=list(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va),y_val),
  callbacks=list(early_stopping,model_checkpoint_def,reduce_lr)
)

## DEF 2.9: Generate Predictions & Invert Scaling ----
val_row_ids<-numW$row_id[-idx]
pred_all<-model%>%predict(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va))
preds_def<-tibble(
  row_id=val_row_ids,
  predicted_total_points_scaled=as.vector(pred_all),
  actual_total_points_scaled=as.vector(y_val)
) %>%
  mutate(
    predicted_total_points=predicted_total_points_scaled*sigma+mu,
    actual_total_points=actual_total_points_scaled*sigma+mu
  ) %>%
  left_join(unscaled_def,by="row_id")

## DEF 2.10 Save model and validation set predictions ----
model_list$def<-model
preds_def_clean<-preds_def%>%select(row_id,GW,player_id,name,position,team,actual_total_points,predicted_total_points,value)

## DEF 2.11: Plots for Evaluation ----
history_def<-history
plot_history_def<-plot(history_def)
print(plot_history_def)

plot_actual_predicted_def<-ggplot(preds_def,aes(x=actual_total_points,y=predicted_total_points))+
  geom_point(alpha=0.5)+
  geom_abline(intercept=0,slope=1,linetype="dashed",color="red")+
  labs(title="DEF Actual vs. Predicted",x="Actual",y="Predicted")+
  facet_wrap(~position)+
  theme_grey()+
  coord_cartesian(xlim=range(preds_def$actual_total_points,na.rm=T),ylim=range(preds_def$predicted_total_points,na.rm=T))
print(plot_actual_predicted_def)

preds_def_residuals<-preds_def%>%mutate(residual=actual_total_points-predicted_total_points)
plot_residuals_predicted_def<-ggplot(preds_def_residuals,aes(x=predicted_total_points,y=residual))+
  geom_point(alpha=0.5)+
  geom_hline(yintercept=0,linetype="dashed",color="red")+
  labs(title="DEF Residuals vs. Predicted",x="Predicted",y="Residual")+
  facet_wrap(~position)+
  theme_grey()
print(plot_residuals_predicted_def)

plot_distribution_residuals_def<-ggplot(preds_def_residuals,aes(x=residual))+
  geom_histogram(aes(y=after_stat(density)),bins=30,fill="blue",alpha=0.7)+
  geom_density(color="red")+
  geom_vline(xintercept=0,linetype="dashed",color="black")+
  labs(title="DEF Residual Distribution",x="Residual",y="Density")+
  facet_wrap(~position)+
  theme_grey()
print(plot_distribution_residuals_def)

model_evaluation_def <-model |> 
  evaluate(c(list(input_seq=Xnum_val), Xcat_hist_va,Xcat_cur_va), y_val, verbose=0)
validation_metrics$def <- model_evaluation_def
validation_metrics$def

# 3: Midfielders (MID) ----

## MID 3.1 Define Features, Target & Scale ----
numF <- c("goals_scored","assists", "creativity", "minutes", "goals_conceded",
          "bonus", "bps", "expected_assists", "expected_goal_involvements",
          "expected_goals", "ict_index", "own_goals", "red_cards",
          "threat", "transfers_in", "transfers_out", "yellow_cards",
          "expected_goals_conceded", "value",
          "selected", "transfers_balance", "starts", "influence",
          "clean_sheets")
tar <- "total_points"
initial_numF_mid <- length(numF)

mid <- mid |> arrange(pID_idx, GW) %>%


mu <- mean(mid[[tar]], na.rm=T)
sigma <- sd(mid[[tar]], na.rm=T)

mid <- mid %>%
  mutate(across(all_of(numF),~(.x-mean(.x,na.rm=T))/sd(.x,na.rm=T)),
         !!tar:=(.data[[tar]]-mu)/sigma)

## MID 3.2: Build rolling numerical windows ----
ws <- vindu
numW <- mid %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(mid,row_id,GW),by="row_id")

## MID 3.3: Lasso Regression ----
Xl <- do.call(rbind,lapply(numW$window,colMeans,na.rm=T))
yl <- numW$target
cvfit <- cv.glmnet(Xl,yl,alpha=1)
minlambda <- cvfit$lambda.min
lassofit <- glmnet(Xl,yl,alpha=1,lambda=minlambda)
keepF <- rownames(as.matrix(coef(lassofit)))[coef(lassofit)[,1]!=0] %>% setdiff("(Intercept)")
numF_mid <- keepF
numF <- keepF

## MID 3.4.1: Rebuild Numeric Windows ----
ws <- vindu
numW <- mid %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(mid,row_id,GW),by="row_id")

### MID 3.4.2: Build & align categorical windows ----
catW_all <- mid %>%
  group_by(player_id) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x%>%select(pID_idx,tID_idx,oID_idx,hID),.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,row_id=rid)
  }) %>%
  ungroup()
catW <- numW %>%
  select(row_id) %>%
  left_join(catW_all,by="row_id")

### MID 3.4.3: To Arrays ----
nSamp<-nrow(numW)
num_array<-array(unlist(numW$window),dim=c(nSamp,ws,length(numF)))
targets<-matrix(numW$target,ncol=1)
pID_hist_array<-t(sapply(catW$window,function(m)m[,"pID_idx"]))
tID_hist_array<-t(sapply(catW$window,function(m)m[,"tID_idx"]))
oID_hist_array<-t(sapply(catW$window,function(m)m[,"oID_idx"]))
hID_hist_array<-t(sapply(catW$window,function(m)m[,"hID"]))
cat_cur_mat<-do.call(rbind,lapply(catW$window,function(m)m[nrow(m),]))
pID_cur_array<-matrix(cat_cur_mat[,"pID_idx"],ncol=1)
tID_cur_array<-matrix(cat_cur_mat[,"tID_idx"],ncol=1)
oID_cur_array<-matrix(cat_cur_mat[,"oID_idx"],ncol=1)
hID_cur_array<-matrix(cat_cur_mat[,"hID"],ncol=1)

## MID 3.5: Split Data ----
idx<-which(numW$GW<=split_gw)
Xnum_train<-num_array[idx,,,drop=F]
Xnum_val<-num_array[-idx,,,drop=F]
y_train<-targets[idx,,drop=F]
y_val<-targets[-idx,,drop=F]
Xcat_hist_tr<-list(pID_hist=pID_hist_array[idx,,drop=F],tID_hist=tID_hist_array[idx,,drop=F],oID_hist=oID_hist_array[idx,,drop=F],hID_hist=hID_hist_array[idx,,drop=F])
Xcat_hist_va<-list(pID_hist=pID_hist_array[-idx,,drop=F],tID_hist=tID_hist_array[-idx,,drop=F],oID_hist=oID_hist_array[-idx,,drop=F],hID_hist=hID_hist_array[-idx,,drop=F])
Xcat_cur_tr<-list(pID_cur=pID_cur_array[idx,,drop=F],tID_cur=tID_cur_array[idx,,drop=F],oID_cur=oID_cur_array[idx,,drop=F],hID_cur=hID_cur_array[idx,,drop=F])
Xcat_cur_va<-list(pID_cur=pID_cur_array[-idx,,drop=F],tID_cur=tID_cur_array[-idx,,drop=F],oID_cur=oID_cur_array[-idx,,drop=F],hID_cur=hID_cur_array[-idx,,drop=F])

## MID 3.6: Building the Keras Neural Network Model ----
nP<-max(mid$pID_idx,na.rm=T);nT<-max(mid$tID_idx,na.rm=T);nO<-max(mid$oID_idx,na.rm=T)
inp_num<-layer_input(shape=c(ws,length(numF)),name="input_seq")
inp_ph<-layer_input(shape=c(ws),dtype="int32",name="pID_hist")
inp_th<-layer_input(shape=c(ws),dtype="int32",name="tID_hist")
inp_oh<-layer_input(shape=c(ws),dtype="int32",name="oID_hist")
inp_hh<-layer_input(shape=c(ws),dtype="int32",name="hID_hist")
inp_pc<-layer_input(shape=c(1),dtype="int32",name="pID_cur")
inp_tc<-layer_input(shape=c(1),dtype="int32",name="tID_cur")
inp_oc<-layer_input(shape=c(1),dtype="int32",name="oID_cur")
inp_hc<-layer_input(shape=c(1),dtype="int32",name="hID_cur")

emb_ph_hist_stream<-inp_ph%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim,mask_zero=T)
emb_th_hist_stream<-inp_th%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim,mask_zero=T)
emb_oh_hist_stream<-inp_oh%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim,mask_zero=T)
emb_hh_hist_stream<-inp_hh%>%layer_lambda(function(x){tf$expand_dims(tf$cast(x,tf$float32),axis=as.integer(-1))})
cat_hist_seq_stream<-layer_concatenate(list(emb_ph_hist_stream,emb_th_hist_stream,emb_oh_hist_stream,emb_hh_hist_stream),axis=-1)

emb_pc_cur_stream<-inp_pc%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim)%>%layer_flatten()
emb_tc_cur_stream<-inp_tc%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim)%>%layer_flatten()
emb_oc_cur_stream<-inp_oc%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim)%>%layer_flatten()
emb_hc_cur_stream<-inp_hc%>%layer_lambda(function(x)tf$cast(x,tf$float32))
seq_pc_cur_stream<-emb_pc_cur_stream%>%layer_repeat_vector(n=ws)
seq_tc_cur_stream<-emb_tc_cur_stream%>%layer_repeat_vector(n=ws)
seq_oc_cur_stream<-emb_oc_cur_stream%>%layer_repeat_vector(n=ws)
seq_hc_cur_stream<-emb_hc_cur_stream%>%layer_repeat_vector(n=ws)
current_cat_as_seq_stream<-layer_concatenate(list(seq_pc_cur_stream,seq_tc_cur_stream,seq_oc_cur_stream,seq_hc_cur_stream),axis=-1)

lstm_in_stream<-layer_concatenate(list(inp_num,cat_hist_seq_stream,current_cat_as_seq_stream),axis=-1)
lstm_out_stream<-lstm_in_stream%>%layer_lstm(units=64,return_sequences=F)%>%layer_dropout(rate=0.2)

head_stream<-lstm_out_stream%>%layer_dense(units=32,activation="relu")
out_stream<-head_stream%>%layer_dense(units=1,activation="linear")

model<-keras_model(inputs=list(inp_num,inp_ph,inp_th,inp_oh,inp_hh,inp_pc,inp_tc,inp_oc,inp_hc),outputs=out_stream,name="mid_lstm_model_v2")
model%>%compile(optimizer=optimizer_adam(),loss="mse",metrics=metric_root_mean_squared_error())

summary(model)
keras_python<-import("keras")
keras_python$utils$plot_model(model,to_file="keras-model-mid-v2.png",show_shapes=T,show_dtype=F,show_layer_names=T,rankdir="TB",expand_nested=F,dpi=200,show_layer_activations=F)
img<-readPNG("keras-model-mid-v2.png")
grid::grid.raster(img)

## MID 3.7: Callback Functions and Scaling Factors ----
weights_filepath_mid<-"R Forecast/best_mid_v2.hdf5"
early_stopping<-callback_early_stopping(monitor="val_loss",patience=num_patience,restore_best_weights=T)
model_checkpoint_mid<-callback_model_checkpoint(filepath=weights_filepath_mid,monitor="val_loss",save_best_only=T,save_weights_only=T,mode="min",verbose=1)
reduce_lr<-callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.2,patience=floor(num_patience*0.5),min_lr=1e-6,verbose=1)
scaling_factors$mid<-list(mu=mu,sigma=sigma,numF=numF)

## MID 3.8: Train Midfielders LSTM Model ----
history<-model%>%fit(
  x=c(list(input_seq=Xnum_train),Xcat_hist_tr,Xcat_cur_tr),
  y=y_train,
  epochs=epoker,
  batch_size=16,
  validation_data=list(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va),y_val),
  callbacks=list(early_stopping,model_checkpoint_mid,reduce_lr)
)

## MID 3.9: Generate Predictions & Invert Scaling ----
val_row_ids<-numW$row_id[-idx]
pred_all<-model%>%predict(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va))
preds_mid<-tibble(
  row_id=val_row_ids,
  predicted_total_points_scaled=as.vector(pred_all),
  actual_total_points_scaled=as.vector(y_val)
) %>%
  mutate(
    predicted_total_points=predicted_total_points_scaled*sigma+mu,
    actual_total_points=actual_total_points_scaled*sigma+mu
  ) %>%
  left_join(unscaled_mid,by="row_id")

## MID 3.10 Save model and validation set predictions ----
model_list$mid<-model
preds_mid_clean<-preds_mid%>%select(row_id,GW,player_id,name,position,team,actual_total_points,predicted_total_points,value)

## MID 3.11: Plots for Evaluation ----
history_mid<-history
plot_history_mid<-plot(history_mid)
print(plot_history_mid)

plot_actual_predicted_mid<-ggplot(preds_mid,aes(x=actual_total_points,y=predicted_total_points))+
  geom_point(alpha=0.5)+
  geom_abline(intercept=0,slope=1,linetype="dashed",color="red")+
  labs(title="MID Actual vs. Predicted",x="Actual",y="Predicted")+
  facet_wrap(~position)+
  theme_grey()+
  coord_cartesian(xlim=range(preds_mid$actual_total_points,na.rm=T),ylim=range(preds_mid$predicted_total_points,na.rm=T))
print(plot_actual_predicted_mid)

preds_mid_residuals<-preds_mid%>%mutate(residual=actual_total_points-predicted_total_points)
plot_residuals_predicted_mid<-ggplot(preds_mid_residuals,aes(x=predicted_total_points,y=residual))+
  geom_point(alpha=0.5)+
  geom_hline(yintercept=0,linetype="dashed",color="red")+
  labs(title="MID Residuals vs. Predicted",x="Predicted",y="Residual")+
  facet_wrap(~position)+
  theme_grey()
print(plot_residuals_predicted_mid)

plot_distribution_residuals_mid<-ggplot(preds_mid_residuals,aes(x=residual))+
  geom_histogram(aes(y=after_stat(density)),bins=30,fill="blue",alpha=0.7)+
  geom_density(color="red")+
  geom_vline(xintercept=0,linetype="dashed",color="black")+
  labs(title="MID Residual Distribution",x="Residual",y="Density")+
  facet_wrap(~position)+
  theme_grey()
print(plot_distribution_residuals_mid)

model_evaluation_mid <- model |>
  evaluate(c(list(input_seq=Xnum_val), Xcat_hist_va,Xcat_cur_va), y_val, verbose=0)

validation_metrics$mid <- model_evaluation_mid


# 4: Forwards (FWD) ----

## FWD 4.1 Define Features, Target & Scale ----
numF <- c("goals_scored","assists", "creativity", "minutes", "goals_conceded",
          "bonus", "bps", "expected_assists", "expected_goal_involvements",
          "expected_goals", "ict_index", "own_goals", "red_cards",
          "threat", "transfers_in", "transfers_out", "yellow_cards",
          "expected_goals_conceded", "value",
          "selected", "transfers_balance", "starts", "influence",
          "clean_sheets")
tar <- "total_points"
initial_numF_fwd <- length(numF) 

fwd <- fwd |> arrange(pID_idx, GW)

mu <- mean(fwd[[tar]], na.rm=T)
sigma <- sd(fwd[[tar]], na.rm=T)

fwd <- fwd %>%
  mutate(across(all_of(numF),~(.x-mean(.x,na.rm=T))/sd(.x,na.rm=T)),
         !!tar:=(.data[[tar]]-mu)/sigma)

## FWD 4.2: Build rolling numerical windows ----
ws <- vindu
numW <- fwd %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(fwd,row_id,GW),by="row_id")

## FWD 4.3: Lasso Regression ----
Xl <- do.call(rbind,lapply(numW$window,colMeans,na.rm=T))
yl <- numW$target
cvfit <- cv.glmnet(Xl,yl,alpha=1)
minlambda <- cvfit$lambda.min
lassofit <- glmnet(Xl,yl,alpha=1,lambda=minlambda)
keepF <- rownames(as.matrix(coef(lassofit)))[coef(lassofit)[,1]!=0] %>% setdiff("(Intercept)")
numF_fwd <- keepF
numF <- keepF

## FWD 4.4.1: Rebuild Numeric Windows ----
ws <- vindu
numW <- fwd %>%
  group_by(pID_idx) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x[,numF],.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    tgt<-.x[[tar]][(ws+1):nrow(.x)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,target=tgt,row_id=rid)
  }) %>%
  ungroup() %>%
  left_join(select(fwd,row_id,GW),by="row_id")

### FWD 4.4.2: Build & align categorical windows ----
catW_all <- fwd %>%
  group_by(player_id) %>%
  filter(n()>ws) %>%
  group_modify(~{
    W<-slide(.x%>%select(pID_idx,tID_idx,oID_idx,hID),.f=~as.matrix(.x),.before=ws,.after=-1,.complete=T)
    valid<-W[(ws+1):length(W)]
    rid<-.x$row_id[(ws+1):nrow(.x)]
    tibble(window=valid,row_id=rid)
  }) %>%
  ungroup()
catW <- numW %>%
  select(row_id) %>%
  left_join(catW_all,by="row_id")

### FWD 4.4.3: To Arrays ----
nSamp<-nrow(numW)
num_array<-array(unlist(numW$window),dim=c(nSamp,ws,length(numF)))
targets<-matrix(numW$target,ncol=1)
pID_hist_array<-t(sapply(catW$window,function(m)m[,"pID_idx"]))
tID_hist_array<-t(sapply(catW$window,function(m)m[,"tID_idx"]))
oID_hist_array<-t(sapply(catW$window,function(m)m[,"oID_idx"]))
hID_hist_array<-t(sapply(catW$window,function(m)m[,"hID"]))
cat_cur_mat<-do.call(rbind,lapply(catW$window,function(m)m[nrow(m),]))
pID_cur_array<-matrix(cat_cur_mat[,"pID_idx"],ncol=1)
tID_cur_array<-matrix(cat_cur_mat[,"tID_idx"],ncol=1)
oID_cur_array<-matrix(cat_cur_mat[,"oID_idx"],ncol=1)
hID_cur_array<-matrix(cat_cur_mat[,"hID"],ncol=1)

## FWD 4.5: Split Data ----
idx<-which(numW$GW<=split_gw)
Xnum_train<-num_array[idx,,,drop=F]
Xnum_val<-num_array[-idx,,,drop=F]
y_train<-targets[idx,,drop=F]
y_val<-targets[-idx,,drop=F]
Xcat_hist_tr<-list(pID_hist=pID_hist_array[idx,,drop=F],tID_hist=tID_hist_array[idx,,drop=F],oID_hist=oID_hist_array[idx,,drop=F],hID_hist=hID_hist_array[idx,,drop=F])
Xcat_hist_va<-list(pID_hist=pID_hist_array[-idx,,drop=F],tID_hist=tID_hist_array[-idx,,drop=F],oID_hist=oID_hist_array[-idx,,drop=F],hID_hist=hID_hist_array[-idx,,drop=F])
Xcat_cur_tr<-list(pID_cur=pID_cur_array[idx,,drop=F],tID_cur=tID_cur_array[idx,,drop=F],oID_cur=oID_cur_array[idx,,drop=F],hID_cur=hID_cur_array[idx,,drop=F])
Xcat_cur_va<-list(pID_cur=pID_cur_array[-idx,,drop=F],tID_cur=tID_cur_array[-idx,,drop=F],oID_cur=oID_cur_array[-idx,,drop=F],hID_cur=hID_cur_array[-idx,,drop=F])

## FWD 4.6: Building the Keras Neural Network Model ----
nP<-max(fwd$pID_idx,na.rm=T);nT<-max(fwd$tID_idx,na.rm=T);nO<-max(fwd$oID_idx,na.rm=T)
inp_num<-layer_input(shape=c(ws,length(numF)),name="input_seq")
inp_ph<-layer_input(shape=c(ws),dtype="int32",name="pID_hist")
inp_th<-layer_input(shape=c(ws),dtype="int32",name="tID_hist")
inp_oh<-layer_input(shape=c(ws),dtype="int32",name="oID_hist")
inp_hh<-layer_input(shape=c(ws),dtype="int32",name="hID_hist")
inp_pc<-layer_input(shape=c(1),dtype="int32",name="pID_cur")
inp_tc<-layer_input(shape=c(1),dtype="int32",name="tID_cur")
inp_oc<-layer_input(shape=c(1),dtype="int32",name="oID_cur")
inp_hc<-layer_input(shape=c(1),dtype="int32",name="hID_cur")

emb_ph_hist_stream<-inp_ph%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim,mask_zero=T)
emb_th_hist_stream<-inp_th%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim,mask_zero=T)
emb_oh_hist_stream<-inp_oh%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim,mask_zero=T)
emb_hh_hist_stream<-inp_hh%>%layer_lambda(function(x){tf$expand_dims(tf$cast(x,tf$float32),axis=as.integer(-1))})
cat_hist_seq_stream<-layer_concatenate(list(emb_ph_hist_stream,emb_th_hist_stream,emb_oh_hist_stream,emb_hh_hist_stream),axis=-1)

emb_pc_cur_stream<-inp_pc%>%layer_embedding(input_dim=nP+1,output_dim=emb_dim)%>%layer_flatten()
emb_tc_cur_stream<-inp_tc%>%layer_embedding(input_dim=nT+1,output_dim=emb_dim)%>%layer_flatten()
emb_oc_cur_stream<-inp_oc%>%layer_embedding(input_dim=nO+1,output_dim=emb_dim)%>%layer_flatten()
emb_hc_cur_stream<-inp_hc%>%layer_lambda(function(x)tf$cast(x,tf$float32))
seq_pc_cur_stream<-emb_pc_cur_stream%>%layer_repeat_vector(n=ws)
seq_tc_cur_stream<-emb_tc_cur_stream%>%layer_repeat_vector(n=ws)
seq_oc_cur_stream<-emb_oc_cur_stream%>%layer_repeat_vector(n=ws)
seq_hc_cur_stream<-emb_hc_cur_stream%>%layer_repeat_vector(n=ws)
current_cat_as_seq_stream<-layer_concatenate(list(seq_pc_cur_stream,seq_tc_cur_stream,seq_oc_cur_stream,seq_hc_cur_stream),axis=-1)

lstm_in_stream<-layer_concatenate(list(inp_num,cat_hist_seq_stream,current_cat_as_seq_stream),axis=-1)
lstm_out_stream<-lstm_in_stream%>%layer_lstm(units=64,return_sequences=F)%>%layer_dropout(rate=0.2)

head_stream<-lstm_out_stream%>%layer_dense(units=32,activation="relu")
out_stream<-head_stream%>%layer_dense(units=1,activation="linear")

model<-keras_model(inputs=list(inp_num,inp_ph,inp_th,inp_oh,inp_hh,inp_pc,inp_tc,inp_oc,inp_hc),outputs=out_stream,name="fwd_lstm_model_v2")
model%>%compile(optimizer=optimizer_adam(),loss="mse",metrics=metric_root_mean_squared_error())

summary(model)
keras_python<-import("keras")
keras_python$utils$plot_model(model,to_file="keras-model-fwd-v2.png",show_shapes=T,show_dtype=F,show_layer_names=T,rankdir="TB",expand_nested=F,dpi=200,show_layer_activations=F)
img<-readPNG("keras-model-fwd-v2.png")
grid::grid.raster(img)

## FWD 4.7: Callback Functions and Scaling Factors ----
weights_filepath_fwd<-"R Forecast/best_fwd_v2.hdf5"
early_stopping<-callback_early_stopping(monitor="val_loss",patience=num_patience,restore_best_weights=T)
model_checkpoint_fwd<-callback_model_checkpoint(filepath=weights_filepath_fwd,monitor="val_loss",save_best_only=T,save_weights_only=T,mode="min",verbose=1)
reduce_lr<-callback_reduce_lr_on_plateau(monitor="val_loss",factor=0.2,patience=floor(num_patience*0.5),min_lr=1e-6,verbose=1)
scaling_factors$fwd<-list(mu=mu,sigma=sigma,numF=numF)

## FWD 4.8: Train Forwards LSTM Model ----
history<-model%>%fit(
  x=c(list(input_seq=Xnum_train),Xcat_hist_tr,Xcat_cur_tr),
  y=y_train,
  epochs=epoker,
  batch_size=16,
  validation_data=list(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va),y_val),
  callbacks=list(early_stopping,model_checkpoint_fwd,reduce_lr)
)

## FWD 4.9: Generate Predictions & Invert Scaling ----
val_row_ids<-numW$row_id[-idx]
pred_all<-model%>%predict(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va))
preds_fwd<-tibble(
  row_id=val_row_ids,
  predicted_total_points_scaled=as.vector(pred_all),
  actual_total_points_scaled=as.vector(y_val)
) %>%
  mutate(
    predicted_total_points=predicted_total_points_scaled*sigma+mu,
    actual_total_points=actual_total_points_scaled*sigma+mu
  ) %>%
  left_join(unscaled_fwd,by="row_id")

## FWD 4.10 Save model and validation set predictions ----
model_list$fwd<-model
preds_fwd_clean<-preds_fwd%>%select(row_id,GW,player_id,name,position,team,actual_total_points,predicted_total_points,value)

## FWD 4.11: Plots for Evaluation ----
history_fwd<-history
plot_history_fwd<-plot(history_fwd)
print(plot_history_fwd)

plot_actual_predicted_fwd<-ggplot(preds_fwd,aes(x=actual_total_points,y=predicted_total_points))+
  geom_point(alpha=0.5)+
  geom_abline(intercept=0,slope=1,linetype="dashed",color="red")+
  labs(title="FWD Actual vs. Predicted",x="Actual",y="Predicted")+
  facet_wrap(~position)+
  theme_grey()+
  coord_cartesian(xlim=range(preds_fwd$actual_total_points,na.rm=T),ylim=range(preds_fwd$predicted_total_points,na.rm=T))
print(plot_actual_predicted_fwd)

preds_fwd_residuals<-preds_fwd%>%mutate(residual=actual_total_points-predicted_total_points)
plot_residuals_predicted_fwd<-ggplot(preds_fwd_residuals,aes(x=predicted_total_points,y=residual))+
  geom_point(alpha=0.5)+
  geom_hline(yintercept=0,linetype="dashed",color="red")+
  labs(title="FWD Residuals vs. Predicted",x="Predicted",y="Residual")+
  facet_wrap(~position)+
  theme_grey()
print(plot_residuals_predicted_fwd)

plot_distribution_residuals_fwd<-ggplot(preds_fwd_residuals,aes(x=residual))+
  geom_histogram(aes(y=after_stat(density)),bins=30,fill="blue",alpha=0.7)+
  geom_density(color="red")+
  geom_vline(xintercept=0,linetype="dashed",color="black")+
  labs(title="FWD Residual Distribution",x="Residual",y="Density")+
  facet_wrap(~position)+
  theme_grey()
print(plot_distribution_residuals_fwd)

model_evaluation_fwd<-model|>evaluate(c(list(input_seq=Xnum_val),Xcat_hist_va,Xcat_cur_va),y_val,verbose=0)
validation_metrics$fwd<-model_evaluation_fwd


# Merging Datasets ----
validation_results_df <- bind_rows(
  preds_gk_clean, preds_def_clean, preds_mid_clean, preds_fwd_clean
) %>% arrange(player_id, GW)

forecastdf_detailed <- bind_rows(preds_gk, preds_def, preds_mid, preds_fwd)

## Evaluation Metrics Dataframe ----
metrics_df <- data.frame(
  metric = names(validation_metrics$mid), # Assumes metric names are consistent
  GK = unlist(validation_metrics$gk),
  DEF = unlist(validation_metrics$def),
  MID = unlist(validation_metrics$mid),
  FWD = unlist(validation_metrics$fwd)
)
validation_metrics
print(metrics_df)

# Save plots
plot_dir <- "C:/Users/peram/Documents/test/R Forecast/Plots from LSTM Forecast script"
if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)

# GK plots
ggsave(file.path(plot_dir,"plot_history_gk_v2.png"), plot=plot_history_gk, width=7, height=5)
ggsave(file.path(plot_dir,"plot_actual_predicted_gk_v2.png"), plot=plot_actual_predicted_gk, width=7, height=5)
ggsave(file.path(plot_dir,"plot_residuals_predicted_gk_v2.png"), plot=plot_residuals_predicted_gk, width=7, height=5)
ggsave(file.path(plot_dir,"plot_distribution_residuals_gk_v2.png"), plot=plot_distribution_residuals_gk, width=7, height=5)

# DEF plots
ggsave(file.path(plot_dir,"plot_history_def_v2.png"), plot=plot_history_def, width=7, height=5)
ggsave(file.path(plot_dir,"plot_actual_predicted_def_v2.png"), plot=plot_actual_predicted_def, width=7, height=5)
ggsave(file.path(plot_dir,"plot_residuals_predicted_def_v2.png"), plot=plot_residuals_predicted_def, width=7, height=5)
ggsave(file.path(plot_dir,"plot_distribution_residuals_def_v2.png"), plot=plot_distribution_residuals_def, width=7, height=5)

# MID plots
ggsave(file.path(plot_dir,"plot_history_mid_v2.png"), plot=plot_history_mid, width=7, height=5)
ggsave(file.path(plot_dir,"plot_actual_predicted_mid_v2.png"), plot=plot_actual_predicted_mid, width=7, height=5)
ggsave(file.path(plot_dir,"plot_residuals_predicted_mid_v2.png"), plot=plot_residuals_predicted_mid, width=7, height=5)
ggsave(file.path(plot_dir,"plot_distribution_residuals_mid_v2.png"), plot=plot_distribution_residuals_mid, width=7, height=5)

# FWD plots
ggsave(file.path(plot_dir,"plot_history_fwd_v2.png"), plot=plot_history_fwd, width=7, height=5)
ggsave(file.path(plot_dir,"plot_actual_predicted_fwd_v2.png"), plot=plot_actual_predicted_fwd, width=7, height=5)
ggsave(file.path(plot_dir,"plot_residuals_predicted_fwd_v2.png"), plot=plot_residuals_predicted_fwd, width=7, height=5)
ggsave(file.path(plot_dir,"plot_distribution_residuals_fwd_v2.png"), plot=plot_distribution_residuals_fwd, width=7, height=5)

## Export Predicted Validation Set as .CSV ----
write_csv(validation_results_df, "Validation_Predictions_Clean_v2.csv")
write_csv(forecastdf_detailed, "Validation_Predictions_Detailed_v2.csv")
write.table(validation_metrics, "Validation metrics.txt")

### Lasso selected features ----
cat("GK Lasso Features:\n"); print(numF_gk) ; length(numF_gk)
cat("DEF Lasso Features:\n"); print(numF_def) ; length(numF_def)
cat("MID Lasso Features:\n"); print(numF_mid) ; length(numF_mid) 
cat("FWD Lasso Features:\n"); print(numF_fwd) ; length(numF_fwd)
initial_numF_gk
initial_numF_def
initial_numF_mid
initial_numF_fwd
