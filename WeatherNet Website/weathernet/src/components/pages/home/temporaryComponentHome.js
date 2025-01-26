import { useNavigate } from "react-router-dom";
import { useEffect } from "react";

export function TemporaryComponentHome() {
  const navigate = useNavigate();
  useEffect(() => {
    navigate(`/`); // Navigate to the home without the /Home path.
  }, []);
}

export default TemporaryComponentHome;
