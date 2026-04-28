library(shiny)

ui <- fluidPage(
  titlePanel("Biopsia virtual - prueba de Render"),
  sidebarLayout(
    sidebarPanel(
      numericInput("edad", "Edad del donante", value = 57, min = 0, max = 100),
      selectInput("sexo", "Sexo", choices = c("Mujer", "Hombre")),
      actionButton("calcular", "Calcular")
    ),
    mainPanel(
      h3("Resultado"),
      verbatimTextOutput("resultado")
    )
  )
)

server <- function(input, output, session) {
  output$resultado <- renderText({
    paste(
      "La aplicación Shiny funciona correctamente.",
      "\nEdad introducida:", input$edad,
      "\nSexo:", input$sexo
    )
  })
}

shinyApp(ui = ui, server = server)