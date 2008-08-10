<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<!-- can't use 3.2 since we can't do without the "archive" element for applets -->
<html>
    <head>
        <title>
            Shephard's Conjecture Play Applet
        </title>
    </head>
    <body bgcolor="#CC9999">

        Shephard's Conjecture Play Applet
        <br>

        <applet
            code="ShephardsPlayApplet.class"
            archive="ShephardsPlayApplet.jar"
            codebase="."
            width="100%" height="90%"
            alt="Your browser understands Java but can't seem to run this applet, sorry."
        >
            <param name="eventVerbose" value="<?php echo $HTTP_GET_VARS["eventVerbose"]; ?>">
            <param name="showControlPanel" value="<?php echo $HTTP_GET_VARS["showControlPanel"]; ?>">
        </applet>

        <br>

        <p>
        Download
        <a href="ShephardsPlayApplet.jar">
        source code
        </a>
        for this applet
        (compressed jar file contains class files and source)

        <hr>
        <table width="100%">  <!-- use full width of page -->
            <tr>
                <td align="left">
                    <?php include("../hitcounter.php"); ?>
                    <br>
                    Last Modified:
Fri Jan 12 12:43:00 PST 2007
                    <address>
                    Don Hatch
                    <br>
                    <a href="mailto:hatch@plunk.org">hatch@plunk.org</a>
                    </address>
                <td align="right">
                        <!--
                        The original:
                            src="http://www.w3.org/Icons/valid-html40"
                        But netscape doesn't do png transparency right, so:
                            src="http://www.w3.org/Icons/valid-html40.gif"
                        -->
                    <a href="http://validator.w3.org/check/referer"><img
                        border="0"
                        src="../images/valid-html40.gif"
                        alt="Valid HTML 4.0!" height="31" width="88"></a>
        </table>

        <center>
            <font size=-1>
                <a href="http://www.plunk.org/~hatch/">
                    Back to Don Hatch's home page.
                </a>
            </font>
        </center>

    </body>
</html>
