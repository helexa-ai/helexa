import { Container, Nav, Navbar } from "react-bootstrap";
import { NavLink, Outlet } from "react-router-dom";

export default function App() {
  return (
    <>
      <Navbar bg="dark" variant="dark" expand="md">
        <Container>
          <Navbar.Brand as={NavLink} to="/">
            helexa&nbsp;bench
          </Navbar.Brand>
          <Nav className="me-auto">
            <Nav.Link as={NavLink} to="/" end>
              Overview
            </Nav.Link>
            <Nav.Link as={NavLink} to="/trends">
              Trends
            </Nav.Link>
            <Nav.Link as={NavLink} to="/runs">
              Runs
            </Nav.Link>
          </Nav>
        </Container>
      </Navbar>
      <Container className="py-4">
        <Outlet />
      </Container>
    </>
  );
}
