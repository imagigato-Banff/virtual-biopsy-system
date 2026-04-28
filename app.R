library(shiny)
library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(stringr)
library(caret)
library(caretEnsemble)
library(randomForest)
library(gbm)
library(xgboost)
library(nnet)
library(MASS)

options(shiny.maxRequestSize = 500 * 1024^2)

model_dir <- "models"
model_paths <- list(
  cv   = file.path(model_dir, "cv_finalround_list_forSynapse.rds"),
  ah   = file.path(model_dir, "ah_finalround_list_forSynapse.rds"),
  IFTA = file.path(model_dir, "IFTA_finalround_list_forSynapse.rds"),
  glo  = file.path(model_dir, "Glo_finalround_list_forSynapse.rds")
)
models_available <- all(file.exists(unlist(model_paths)))
loaded_models <- NULL
if (models_available) {
  loaded_models <- list(
    cv   = readRDS(model_paths$cv),
    ah   = readRDS(model_paths$ah),
    IFTA = readRDS(model_paths$IFTA),
    glo  = readRDS(model_paths$glo)
  )
}

# ========= Núcleo: predicción con los caret::train reales, sin recorrer subobjetos internos =========

is_valid_train <- function(x) {
  inherits(x, "train") && !is.null(x$finalModel)
}

extract_train_models <- function(container, regression = FALSE) {
  keep_names <- if (regression) {
    c("rf", "gbm", "xgbTree", "avNNet")
  } else {
    c("rf", "gbm", "xgbTree", "lda", "avNNet", "mnom", "multinom")
  }

  out <- list()

  if (is_valid_train(container)) {
    out$model <- container
    return(out)
  }

  if (is.list(container)) {
    nms <- names(container)
    if (!is.null(nms)) {
      for (nm in intersect(keep_names, nms)) {
        if (is_valid_train(container[[nm]])) out[[nm]] <- container[[nm]]
      }
    }

    for (slot in c("model", "models", "caretList", "model_list", "list")) {
      if (slot %in% names(container) && is.list(container[[slot]])) {
        nms2 <- names(container[[slot]])
        if (!is.null(nms2)) {
          for (nm in intersect(keep_names, nms2)) {
            if (is_valid_train(container[[slot]][[nm]])) out[[nm]] <- container[[slot]][[nm]]
          }
        }
      }
    }
  }

  out
}

patient_values <- function(input) {
  deceased <- input$donor_type == "Fallecido"
  sex_female <- input$sex == "Mujer"
  yes <- function(x) identical(x, "Sí")

  list(
    Age = as.numeric(input$age),
    Gender = as.integer(sex_female),
    Gender1 = as.integer(sex_female),
    Sex = as.integer(sex_female),
    Sex1 = as.integer(sex_female),
    Donor_type = as.integer(deceased),
    Donor_type1 = as.integer(deceased),
    Hypertension = as.integer(yes(input$hypertension)),
    Hypertension1 = as.integer(yes(input$hypertension)),
    Diabetes = as.integer(yes(input$diabetes)),
    Diabetes1 = as.integer(yes(input$diabetes)),
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = as.integer(yes(input$proteinuria)),
    Proteinuria1 = as.integer(yes(input$proteinuria)),
    HCV_status = as.integer(yes(input$hcv)),
    HCV_status1 = as.integer(yes(input$hcv)),
    DCD = as.integer(deceased && yes(input$dcd)),
    DCD1 = as.integer(deceased && yes(input$dcd)),
    bmi = as.numeric(input$bmi),
    BMI = as.numeric(input$bmi),
    vascular_death = as.integer(deceased && yes(input$vascular_death)),
    vascular_death1 = as.integer(deceased && yes(input$vascular_death))
  )
}

resolve_value <- function(nm, vals) {
  if (nm %in% names(vals)) return(vals[[nm]])
  if (grepl("1$", nm)) {
    base <- sub("1$", "", nm)
    if (base %in% names(vals)) return(vals[[base]])
  }
  alt <- paste0(nm, "1")
  if (alt %in% names(vals)) return(vals[[alt]])
  NA
}

pick_level <- function(var, value, levs) {
  if (length(levs) == 0) return(as.character(value))
  levs_chr <- as.character(levs)
  low <- tolower(levs_chr)
  value <- as.integer(value)

  if (var %in% c("Gender", "Gender1", "Sex", "Sex1")) {
    if (value == 1) {
      hit <- grep("female|woman|mujer|femenino|^f$|^1$|yes|true", low)
      if (length(hit)) return(levs_chr[hit[1]])
    } else {
      hit <- grep("male|man|hombre|masculino|^m$|^0$|no|false", low)
      if (length(hit)) return(levs_chr[hit[1]])
    }
  }

  if (var %in% c("Donor_type", "Donor_type1")) {
    if (value == 1) {
      hit <- grep("deceased|fallecido|cadaver|dead|^1$|yes|true", low)
      if (length(hit)) return(levs_chr[hit[1]])
    } else {
      hit <- grep("living|vivo|live|^0$|no|false", low)
      if (length(hit)) return(levs_chr[hit[1]])
    }
  }

  if (value == 1) {
    hit <- grep("^1$|yes|si|sí|true|positive|pos", low)
    if (length(hit)) return(levs_chr[hit[1]])
  }
  if (value == 0) {
    hit <- grep("^0$|no|false|negative|neg", low)
    if (length(hit)) return(levs_chr[hit[1]])
  }
  if (length(levs_chr) == 2) return(if (value == 1) levs_chr[2] else levs_chr[1])
  levs_chr[1]
}

predictor_names_for_train <- function(m) {
  if (!is.null(m$terms)) {
    return(all.vars(delete.response(m$terms)))
  }
  if (!is.null(m$trainingData)) {
    return(setdiff(names(m$trainingData), ".outcome"))
  }
  if (!is.null(m$coefnames)) return(m$coefnames)
  character(0)
}

classes_for_train <- function(m) {
  out <- list()
  if (!is.null(m$terms)) {
    dc <- attr(m$terms, "dataClasses")
    if (!is.null(dc)) {
      dc <- dc[names(dc) != ".outcome"]
      out <- as.list(dc)
    }
  }
  if (!length(out) && !is.null(m$trainingData)) {
    for (nm in setdiff(names(m$trainingData), ".outcome")) {
      out[[nm]] <- class(m$trainingData[[nm]])[1]
    }
  }
  out
}

levels_for_var <- function(m, nm) {
  if (!is.null(m$xlevels) && nm %in% names(m$xlevels)) return(m$xlevels[[nm]])
  if (!is.null(m$finalModel$xlevels) && nm %in% names(m$finalModel$xlevels)) return(m$finalModel$xlevels[[nm]])
  if (!is.null(m$trainingData) && nm %in% names(m$trainingData) && is.factor(m$trainingData[[nm]])) return(levels(m$trainingData[[nm]]))
  NULL
}

make_newdata <- function(m, vals) {
  pred_names <- predictor_names_for_train(m)
  cls <- classes_for_train(m)
  if (!length(pred_names)) stop("No se pudieron leer los predictores del modelo.")

  nd <- vector("list", length(pred_names))
  names(nd) <- pred_names

  for (nm in pred_names) {
    v <- resolve_value(nm, vals)
    expected <- if (nm %in% names(cls)) as.character(cls[[nm]]) else "numeric"
    levs <- levels_for_var(m, nm)

    if (expected %in% c("factor", "ordered")) {
      chosen <- pick_level(nm, v, levs)
      if (expected == "ordered") nd[[nm]] <- ordered(chosen, levels = levs) else nd[[nm]] <- factor(chosen, levels = levs)
    } else if (expected %in% c("logical")) {
      nd[[nm]] <- as.logical(as.integer(v))
    } else if (expected %in% c("integer")) {
      nd[[nm]] <- as.integer(v)
    } else {
      nd[[nm]] <- as.numeric(v)
    }
  }

  as.data.frame(nd, check.names = FALSE)
}

normalise_probs <- function(p) {
  p <- as.data.frame(p, check.names = FALSE)
  nm <- names(p)
  nm <- gsub("^X", "", nm)
  nm <- gsub("[^0-9]", "", nm)
  names(p) <- nm
  for (k in as.character(0:3)) if (!k %in% names(p)) p[[k]] <- 0
  p <- p[, as.character(0:3), drop = FALSE]
  p <- as.data.frame(lapply(p, as.numeric), check.names = FALSE)
  p[is.na(p)] <- 0
  s <- rowSums(p)
  if (s[1] <= 0) p[,] <- 0.25 else p <- p / s
  names(p) <- paste0("X", 0:3)
  p[1, , drop = FALSE]
}

predict_probs <- function(container, vals) {
  models <- extract_train_models(container, regression = FALSE)
  if (!length(models)) stop("Los RDS cargan, pero no contienen modelos caret::train predictivos utilizables.")

  probs <- list()
  errors <- character()
  for (nm in names(models)) {
    m <- models[[nm]]
    nd <- tryCatch(make_newdata(m, vals), error = function(e) { errors <<- c(errors, paste0(nm, ": ", conditionMessage(e))); NULL })
    if (is.null(nd)) next
    pr <- tryCatch(predict(m, newdata = nd, type = "prob"), error = function(e) { errors <<- c(errors, paste0(nm, ": ", conditionMessage(e))); NULL })
    if (!is.null(pr)) {
      probs[[length(probs) + 1]] <- normalise_probs(pr)
    }
  }
  if (!length(probs)) stop(paste("No se pudieron calcular probabilidades reales.", paste(head(errors, 6), collapse = " | ")))
  arr <- simplify2array(lapply(probs, as.matrix))
  normalise_probs(as.data.frame(apply(arr, c(1, 2), mean), check.names = FALSE))
}

predict_glo <- function(container, vals) {
  models <- extract_train_models(container, regression = TRUE)
  if (!length(models)) stop("El RDS de glomeruloesclerosis no contiene modelos caret::train predictivos utilizables.")

  pred <- numeric(0)
  errors <- character()
  for (nm in names(models)) {
    m <- models[[nm]]
    nd <- tryCatch(make_newdata(m, vals), error = function(e) { errors <<- c(errors, paste0(nm, ": ", conditionMessage(e))); NULL })
    if (is.null(nd)) next
    pr <- tryCatch(predict(m, newdata = nd), error = function(e) { errors <<- c(errors, paste0(nm, ": ", conditionMessage(e))); NULL })
    if (!is.null(pr)) {
      v <- suppressWarnings(as.numeric(pr[1]))
      if (!is.na(v)) pred <- c(pred, v)
    }
  }
  if (!length(pred)) stop(paste("No se pudo calcular glomeruloesclerosis.", paste(head(errors, 6), collapse = " | ")))
  max(0, min(100, mean(pred)))
}

pred_class <- function(p) which.max(as.numeric(p[1, paste0("X", 0:3)])) - 1
prob_ge2 <- function(p) as.numeric(p$X2 + p$X3)
lesion_name <- function(x) switch(x, cv = "Arteriosclerosis (cv)", ah = "Hialinosis arteriolar (ah)", IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)")
grade_label <- function(x) c("0 - ausente", "1 - leve", "2 - moderada", "3 - grave")[[as.integer(x) + 1]]

clinical_text <- function(probs, glo) {
  out <- character()
  for (nm in names(probs)) {
    p <- probs[[nm]]; cls <- pred_class(p); sev <- prob_ge2(p)
    if (sev >= 0.50 || cls >= 2) {
      out <- c(out, paste0(lesion_name(nm), ": alta probabilidad de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Considerar seguimiento estrecho y contextualización con biopsias posteriores."))
    } else if (sev >= 0.25) {
      out <- c(out, paste0(lesion_name(nm), ": probabilidad intermedia de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Interpretar junto con edad, comorbilidad y función renal del donante."))
    } else {
      out <- c(out, paste0(lesion_name(nm), ": predominan grados ausente o leve. Clase predicha: ", grade_label(cls), "."))
    }
  }
  if (glo >= 20) out <- c(out, paste0("Glomeruloesclerosis estimada elevada: ", round(glo, 1), "%."))
  else if (glo >= 10) out <- c(out, paste0("Glomeruloesclerosis estimada intermedia: ", round(glo, 1), "%."))
  else out <- c(out, paste0("Glomeruloesclerosis estimada baja: ", round(glo, 1), "%."))
  paste(out, collapse = "\n\n")
}

ui <- fluidPage(
  tags$head(tags$style(HTML("body{background:#f6f8fb;color:#1f2937}.main-title{background:linear-gradient(135deg,#19324a,#295c7a);color:white;padding:22px;border-radius:16px;margin-bottom:20px}.main-title h1{margin-top:0;font-weight:700}.panel-card{background:white;border-radius:16px;padding:18px;box-shadow:0 2px 10px rgba(0,0,0,.08);margin-bottom:16px}.warning-card{background:#fff3cd;border-left:6px solid #f0ad4e;padding:14px;border-radius:10px;margin-bottom:16px}.ok-card{background:#eaf7ee;border-left:6px solid #2e8b57;padding:14px;border-radius:10px;margin-bottom:16px}.result-number{font-size:30px;font-weight:700;color:#19324a}.small-muted{color:#6b7280;font-size:13px}h3{font-weight:700;color:#19324a}.error-box{color:#b00020;background:#fff5f5;border-left:5px solid #b00020;padding:12px;border-radius:8px;white-space:pre-wrap}"))),
  div(class="main-title", h1("The Virtual Biopsy System"), h4("Biopsia virtual día cero para trasplante renal"), p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")),
  if (!models_available) div(class="warning-card", h4("Modelos no encontrados"), p("Faltan archivos .rds en models/.")) else div(class="ok-card", strong("Modelos cargados correctamente. "), span("La aplicación está lista para calcular predicciones.")),
  sidebarLayout(
    sidebarPanel(width=4, div(class="panel-card", h3("Datos del donante"),
      numericInput("age", "Edad del donante (años)", 57, min=0, max=100, step=1),
      selectInput("sex", "Sexo", c("Mujer", "Hombre"), selected="Mujer"),
      selectInput("donor_type", "Tipo de donante", c("Vivo", "Fallecido"), selected="Fallecido"),
      conditionalPanel("input.donor_type == 'Fallecido'", selectInput("vascular_death", "Causa de muerte cerebrovascular", c("No", "Sí"), selected="Sí"), selectInput("dcd", "Causa de muerte circulatoria / DCD", c("No", "Sí"), selected="No")),
      selectInput("hypertension", "Hipertensión", c("No", "Sí"), selected="No"),
      selectInput("diabetes", "Diabetes mellitus", c("No", "Sí"), selected="No"),
      selectInput("hcv", "Estado VHC", c("No", "Sí"), selected="No"),
      numericInput("bmi", "Índice de masa corporal / BMI (kg/m²)", 20, min=10, max=70, step=.1),
      numericInput("creatinine", "Creatinina sérica más baja (mg/dL)", .6, min=.1, max=15, step=.1),
      selectInput("proteinuria", "Proteinuria", c("No", "Sí"), selected="No"),
      p(class="small-muted", "Proteinuria positiva: tira reactiva ≥1 o UPCR ≥0.5 g/g."),
      actionButton("calculate", "Calcular biopsia virtual", class="btn-primary", width="100%")
    )),
    mainPanel(width=8, tabsetPanel(
      tabPanel("Resultados", br(), div(class="panel-card", h3("Resumen de predicción"), uiOutput("summary_cards")), div(class="panel-card", h3("Probabilidades por lesión y grado Banff"), DTOutput("prob_table")), div(class="panel-card", h3("Glomeruloesclerosis"), htmlOutput("glo_output"))),
      tabPanel("Gráfico radar", br(), div(class="panel-card", h3("Radar de probabilidades"), plotlyOutput("radar_plot", height="560px"))),
      tabPanel("Nota clínica", br(), div(class="panel-card", h3("Interpretación clínica automática"), verbatimTextOutput("clinical_note"))),
      tabPanel("Ayuda", br(), div(class="panel-card", h3("Qué calcula esta aplicación"), p("Probabilidades Banff 0-3 para cv, ah e IFTA, y porcentaje de glomeruloesclerosis."), p("Herramienta de investigación; no sustituye la valoración clínica.")))
    ))
  )
)

server <- function(input, output, session) {
  prediction <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan los archivos .rds en models/."))
    vals <- patient_values(input)
    probs <- list(cv=predict_probs(loaded_models$cv, vals), ah=predict_probs(loaded_models$ah, vals), IFTA=predict_probs(loaded_models$IFTA, vals))
    glo <- predict_glo(loaded_models$glo, vals)
    list(vals=vals, probs=probs, glo=glo)
  }, ignoreInit=FALSE)

  output$summary_cards <- renderUI({
    r <- prediction()
    tagList(lapply(names(r$probs), function(nm) {
      p <- r$probs[[nm]]; cls <- pred_class(p); sev <- prob_ge2(p)
      div(style="border-bottom:1px solid #e5e7eb;padding:12px 0;", h4(lesion_name(nm)), p(strong("Clase predicha: "), grade_label(cls)), p(strong("Probabilidad de grado ≥2: "), paste0(round(100*sev,1), "%")))
    }))
  })

  output$prob_table <- renderDT({
    r <- prediction()
    tab <- imap_dfr(r$probs, ~tibble(Lesión=lesion_name(.y), `Grado 0 (%)`=round(100*.x$X0,1), `Grado 1 (%)`=round(100*.x$X1,1), `Grado 2 (%)`=round(100*.x$X2,1), `Grado 3 (%)`=round(100*.x$X3,1), `Clase predicha`=grade_label(pred_class(.x)), `Probabilidad grado ≥2 (%)`=round(100*prob_ge2(.x),1)))
    datatable(tab, rownames=FALSE, options=list(dom="t", pageLength=3))
  })

  output$glo_output <- renderUI({
    r <- prediction(); HTML(paste0("<div class='result-number'>", round(r$glo,2), "%</div><p>Porcentaje estimado de glomeruloesclerosis.</p>"))
  })

  output$clinical_note <- renderText({
    r <- prediction(); clinical_text(r$probs, r$glo)
  })

  output$radar_plot <- renderPlotly({
    r <- prediction()
    dat <- imap_dfr(r$probs, function(p, nm) tibble(lesion=lesion_name(nm), grado=factor(c("Grado 0","Grado 1","Grado 2","Grado 3"), levels=c("Grado 0","Grado 1","Grado 2","Grado 3")), prob=as.numeric(p[1,paste0("X",0:3)])))
    plot_ly(type="scatterpolar", mode="lines+markers") %>%
      add_trace(data=dat[dat$lesion=="Arteriosclerosis (cv)",], r=~prob, theta=~grado, name="cv", fill="toself") %>%
      add_trace(data=dat[dat$lesion=="Hialinosis arteriolar (ah)",], r=~prob, theta=~grado, name="ah", fill="toself") %>%
      add_trace(data=dat[dat$lesion=="Fibrosis intersticial y atrofia tubular (IFTA)",], r=~prob, theta=~grado, name="IFTA", fill="toself") %>%
      layout(polar=list(radialaxis=list(visible=TRUE, range=c(0,1), tickformat=".0%")), showlegend=TRUE, title="Probabilidades predichas por grado Banff")
  })
}

shinyApp(ui=ui, server=server)
