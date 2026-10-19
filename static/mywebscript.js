let showResponse = (message, isError) => {
    let output = document.getElementById("system_response");
    output.textContent = message;
    output.className = isError ? "response-error" : "response-success";
};

let RunSentimentAnalysis = () => {
    let textToAnalyze = document.getElementById("textToAnalyze").value;

    let xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState === 4) {
            showResponse(this.responseText, this.status !== 200);
        }
    };
    xhttp.open("GET", "emotionDetector?textToAnalyze=" + encodeURIComponent(textToAnalyze), true);
    xhttp.send();
};

window.addEventListener("DOMContentLoaded", () => {
    let params = new URLSearchParams(window.location.search);
    if (params.has("textToAnalyze")) {
        document.getElementById("textToAnalyze").value = params.get("textToAnalyze");
        RunSentimentAnalysis();
    }
});
