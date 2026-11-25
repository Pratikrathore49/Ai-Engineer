
import { BrowserRouter } from 'react-router-dom'
import AppRouter from "./AppRouter"
import { Toaster } from 'sonner'


function App() {
    return (<>    
    <Toaster position='top-right' richColors />
        <BrowserRouter>
        <AppRouter />
    </BrowserRouter></>

    )
}

export default App
