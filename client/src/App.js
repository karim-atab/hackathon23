import React, {useState, useEffect} from 'react'
<script src="https://unpkg.com/axios/dist/axios.min.js"></script>

function App(){

  const [data, setData] = useState([{}])
  const [dataa, setDataa] = useState([{}])
  
  useEffect(() => {
    fetch("/categories").then(
      res => res.json()
    ).then(
      data => {
        setData(data)
        console.log(data)
      }
    )
  }, [])

  const prediction = () => {
    const complaintText = document.getElementById('idd').value
    fetch('/makepred',{
      method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(complaintText),
    })
    .then(res => res.json()
    ).then(
      dataa => {
        setDataa(dataa)
        
        document.getElementById('predd').innerText = JSON.stringify(dataa)
        console.log("test",dataa)
      }
    )
  }
  function getText(){
    prediction()

  }

  return(
    <div>
      <body>
        <textarea id='idd'>
        </textarea>
        <button onClick={getText}>click me</button>
        <p id='predd'></p>
      </body>
    </div>
  )
}

export default App