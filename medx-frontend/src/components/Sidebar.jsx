import { Link } from "react-router-dom";

export default function Sidebar(){

return(

<div className="sidebar">

<h2>MedX AI</h2>

<Link to="/">Dashboard</Link>

<Link to="/register">Register</Link>

<Link to="/prescription">Prescription</Link>

<Link to="/history">History</Link>

</div>

);

}